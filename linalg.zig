const std = @import("std");

// Define type shortcuts.
pub const Vector2f = @Vector(2, f32);
pub const Vector3f = @Vector(3, f32);
pub const Vector4f = @Vector(4, f32);
pub const Quaternionf = Vector4f;

pub const Vector2d = @Vector(2, f64);
pub const Vector3d = @Vector(3, f64);
pub const Vector4d = @Vector(4, f64);
pub const Quaterniond = Vector4d;

pub const Matrixf = [4]Vector4f;
pub const Matrixd = [4]Vector4d;

pub inline fn checkType(T: type) void {
    switch (@typeInfo(T)) {
        .vector => |v| if (v.child != f32 and v.child != f64) @compileError("This library only works with float type vectors at this time. Got type: " ++ @typeName(v.child)),
        else => @compileError("Only Vector types are accepted"),
    }
}
pub inline fn checkLen(T: type, expected: comptime_int) void {
    switch (@typeInfo(T)) {
        .vector => |v| if (v.child != f32 and v.child != f64) {
            @compileError("This library only works with float type vectors at this time. Got Type: " ++ @typeName(v.child));
        } else if (!(v.len == expected)) {
            @compileError("Vector of incorrect length passed. Expected: " ++ expected ++ "\tGot: " ++ v.len ++ "\n");
        },
        else => @compileError("Only Vector types are accepted"),
    }
}
pub inline fn zero(T: type) T {
    return @as(T, @splat(0));
}
pub inline fn one(T: type) T {
    return @as(T, @splat(1));
}

// Swizzle function (from https://github.com/johan0A/zig-linear-algebra/blob/main/src/vector.zig)
pub fn info(T: type) std.builtin.Type.Vector {
    if (@typeInfo(T) != .vector) @compileError("Expected a @Vector type got: " ++ @typeName(T));
    return @typeInfo(T).vector;
}
fn vecLen(T: type) comptime_int {
    if (@typeInfo(T) != .vector) @compileError("Expected a @Vector type got: " ++ @typeName(T));
    return @typeInfo(T).vector.len;
}
pub fn sw(vec: anytype, comptime components: []const u8) @Vector(components.len, info(@TypeOf(vec)).child) {
    const T = info(@TypeOf(vec)).child;
    comptime var mask: [components.len]u8 = undefined;
    comptime var i: usize = 0;
    inline for (components) |c| {
        switch (c) {
            'x' => mask[i] = 0,
            'y' => mask[i] = 1,
            'z' => mask[i] = 2,
            'w' => mask[i] = 3,
            else => @compileError("swizzle: invalid component"),
        }
        i += 1;
    }

    return @shuffle(
        T,
        vec,
        @as(@Vector(1, T), undefined),
        mask,
    );
}

// Generic functions for all Vecor types.
pub inline fn negate(v: anytype) @TypeOf(v) {
    checkType(@TypeOf(v));
    return -v;
}
pub inline fn addScalar(v: anytype, s: @TypeOf(v[0])) @TypeOf(v) {
    const T = @TypeOf(v);
    checkType(T);
    return v + @as(T, @splat(s));
}
pub inline fn subScalar(v: anytype, s: @TypeOf(v[0])) @TypeOf(v) {
    const T = @TypeOf(v);
    checkType(T);
    return v - @as(T, @splat(s));
}
pub inline fn mulScalar(v: anytype, s: @TypeOf(v[0])) @TypeOf(v) {
    const T = @TypeOf(v);
    checkType(T);
    return v * @as(T, @splat(s));
}
pub inline fn divScalar(v: anytype, s: @TypeOf(v[0])) @TypeOf(v) {
    const T = @TypeOf(v);
    checkType(T);
    if (s == 0) return zero(T);
    return v / @as(T, @splat(s));
}
pub inline fn dot(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1[0]) {
    checkType(@TypeOf(v1));
    return @reduce(.Add, v1 * v2);
}
pub inline fn cross(v1: anytype, v2: @TypeOf(v1)) if (info(@TypeOf(v1)).len == 2) @TypeOf(v1[0]) else @TypeOf(v1) {
    switch (info(@TypeOf(v1)).len) {
        2 => return vector2Cross(v1, v2),
        3 => return vector3Cross(v1, v2),
        // 4 => return vector4Cross(v1, v2),
        else => @compileError("Cross Product can only handle 2, 3, 4 length vectors"),
    }
}
pub inline fn lenSqr(v: anytype) @TypeOf(v[0]) {
    return dot(v, v);
}
pub inline fn len(v: anytype) @TypeOf(v[0]) {
    const length = lenSqr(v);
    if (length == 0) return 0;
    return @sqrt(length);
}
pub inline fn normalize(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    checkType(T);
    return v / @as(T, @splat(if (len(v) == 0) 1 else len(v)));
}
pub inline fn distance(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1[0]) {
    const dist = distanceSqr(v1, v2);
    if (dist == 0) return 0;
    return @sqrt(dist);
}
pub inline fn distanceSqr(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1[0]) {
    checkType(@TypeOf(v1));
    return lenSqr(v1 - v2);
}
pub inline fn angle(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1[0]) {
    switch (info(@TypeOf(v1)).len) {
        2 => return vector2Angle(v1, v2),
        3 => return vector3Angle(v1, v2),
        // 4 => return vector4Cross(v1, v2),
        else => @compileError("Cross Product can only handle 2, 3, 4 length vectors"),
    }
}
pub inline fn lerp(v1: anytype, v2: @TypeOf(v1), amount: @TypeOf(v1[0])) @TypeOf(v1) {
    checkType(@TypeOf(v1));
    return v1 + mulScalar(v2 - v1, amount);
}
pub inline fn reflect(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1) {
    checkType(@TypeOf(v1));
    return v1 + mulScalar(mulScalar(v2, 2), dot(v1, v2));
}
pub inline fn invert(v: anytype) @TypeOf(v) {
    return one(@TypeOf(v)) / v;
}
pub inline fn clamp(v: anytype, min: @TypeOf(v), max: @TypeOf(v)) @TypeOf(v) {
    return @min(max, @max(v, min));
}
pub inline fn clampLength(v: anytype, min: @TypeOf(v[0]), max: @TypeOf(v[0])) @TypeOf(v) {
    var length = lenSqr(v);
    if (length == 0) return zero(@TypeOf(v));
    length = @sqrt(length);
    return mulScalar(v, if (length < min) min / length else if (length > max) max / length else 1.0);
}
pub inline fn equals(v1: anytype, v2: @TypeOf(v1)) bool {
    return @abs(v1 - v2) <= mulScalar(@max(one(@TypeOf(v1)), @max(@abs(v1), @abs(v2))), std.math.floatEps(@TypeOf(v1[0])));
}
pub inline fn refract(v: anytype, n: @TypeOf(v), r: @TypeOf(v[0])) @TypeOf(v) {
    const norm = normalize(n);
    const dt = dot(v, norm);
    var d = 1 - (r * r * (1 - (dt * dt)));
    if (d < 0) return zero(@TypeOf(v));
    d = @sqrt(d);
    return v * mulScalar(v, r) - mulScalar(n, r * dot + d);
}
pub inline fn transform(v: anytype, m: anytype) @TypeOf(v) {
    // Verify m is a matrix.
    std.debug.print("Type of V: {any}\n", .{@typeInfo(@TypeOf(v))});
    std.debug.print("Type of M: {any}\n", .{@typeInfo(@TypeOf(m))});
    // First things first, make sure m is an array of vectors of length 4.
    const mInfo = @typeInfo(@TypeOf(m));
    switch (mInfo) {
        .array => {
            if (mInfo.array.len != 4) @compileError("m is expected to be a [4]Vector4{f/d}");
            const mcInfo = @typeInfo(mInfo.array.child);
            switch (mcInfo) {
                .vector => if (mcInfo.vector.len != 4 or (mcInfo.vector.child != f32 and mcInfo.vector.child != f64)) @compileError("m is expected to be a [4]Vector4{f/d}"),
                else => @compileError("m is expected to be a [4]Vector4{f/d}"),
            }
        },
        else => @compileError("m is expected to be a [4]Vector4{f/d}"),
    }
    // Now check that v and m child types match.
    if (@TypeOf(v[0]) != @TypeOf(m[0][0])) @compileError("Vector and Matrix need to have the same float type.");

    var length: usize = 2;
    // Get the vector length.
    const vInfo = @typeInfo(@TypeOf(v));
    switch (vInfo) {
        .vector => if (vInfo.vector.len != 2 and vInfo.vector.len != 3) {
            @compileError("V is expected to be a vector2 or vector3");
        } else {
            length = vInfo.vector.len;
        },
        else => @compileError("V is expected to be a vector2 or vector3"),
    }

    // First, invert the matrix across it's diagonal, just to make the math faster.
    const mi = matrixTranspose(m);

    var out = zero(@TypeOf(v));
    // Expand v
    var v_exp = zero(@Vector(4, @TypeOf(v[0])));
    v_exp[0] = v[0];
    v_exp[1] = v[1];
    // OK, we can transform.
    inline for (0..length) |i| {
        out[i] = @reduce(.Add, v_exp * mi[i]) + m[3][i];
    }
    return out;
}
pub inline fn moveTowards(v: anytype, tgt: anytype, maxDistance: @TypeOf(v[0])) @TypeOf(v) {
    const dVec = tgt - v;
    const dLen = lenSqr(dVec);
    if ((dLen == 0) or ((maxDistance >= 0) and (dLen <= (maxDistance * maxDistance)))) return tgt;
    return v + mulScalar(divScalar(dVec, @sqrt(dLen)), maxDistance);
}

// Vector2 specific functions.
inline fn vector2Cross(v1: anytype, v2: anytype) @TypeOf(v1[0]) {
    checkLen(@TypeOf(v1), 2);
    return dot(v1, vector2RotL(v2));
}
pub inline fn vector2RotL(v: anytype) @TypeOf(v) {
    checkLen(@TypeOf(v), 2);
    return @TypeOf(v){ -v[1], v[0] };
}
pub inline fn vector2RotR(v: anytype) @TypeOf(v) {
    checkLen(@TypeOf(v), 2);
    return @TypeOf(v){ v[1], -v[0] };
}
inline fn vector2Angle(v1: anytype, v2: anytype) @TypeOf(v1[0]) {
    checkLen(@TypeOf(v1), 2);
    return std.math.atan2(cross(v1, v2), dot(v1, v2));
}
pub inline fn vector2LineAngle(v1: anytype, v2: anytype) @TypeOf(v1[0]) {
    checkLen(@TypeOf(v1), 2);
    // Relies on normalized vectors, so normalize them.
    const n1 = normalize(v1);
    const n2 = normalize(v2);
    const ang = n2 - n1;
    return std.math.atan2(ang[1], ang[0]);
}
pub inline fn vector2RotateRadians(v: anytype, rad: @TypeOf(v[0])) @TypeOf(v) {
    checkLen(@TypeOf(v), 2);
    const rVec = @TypeOf(v){ @cos(rad), @sin(rad) };
    return vector2Rotate(v, rVec);
}
pub inline fn vector2Rotate(v: anytype, rVec: anytype) @TypeOf(v) {
    checkLen(@TypeOf(v), 2);
    return vector2Cross(v, rVec);
}

// Vector 3 specific
pub inline fn vector3Cross(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1) {
    checkLen(@TypeOf(v1), 3);
    return @TypeOf(v1){ v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0] };
}
inline fn vector3Angle(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1[0]) {
    checkLen(@TypeOf(v1), 3);
    return std.math.atan2(@sqrt(dot(cross(v1, v2))), dot(v1, v2));
}
pub inline fn vector3Project(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1) {
    checkLen(@TypeOf(v1), 3);
    const div = lenSqr(v2);
    if (div == 0) return zero(@TypeOf(v1));
    return mulScalar(v2, dot(v1, v2) / div);
}
pub inline fn vector3Reject(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1) {
    checkLen(@TypeOf(v1), 3);
    const div = lenSqr(v2);
    if (div == 0) return zero(@TypeOf(v1));
    return v1 - mulScalar(v2, dot(v1, v2) / div);
}
pub inline fn vector3Perpendicular(v: anytype) @TypeOf(v) {
    checkLen(@TypeOf(v), 3);
    const minVec = @abs(v);
    const min = @reduce(.Min, v);
    var cardinalAxis = zero(@TypeOf(v));
    inline for (0..3) |i| {
        if (min == minVec[i]) {
            cardinalAxis[i] = 1;
            break;
        }
    }
    return cross(v, cardinalAxis);
}
pub inline fn vector3OrthoNormalize(v1: anytype, v2: @TypeOf(v1)) [2]@TypeOf(v1) {
    checkLen(@TypeOf(v1), 3);
    // Get the raw data for vp1 and vp2.
    const vout1 = normalize(v1);
    var vn1 = cross(v1, v2);
    vn1 = normalize(vn1);
    const vn2 = cross(vn1, vout1);
    return [2]@TypeOf(v1){ vout1, vn2 };
}
pub inline fn vector3RotateQuaternion(v: anytype, q: anytype) @TypeOf(v) {
    checkLen(@TypeOf(v), 3);
    checkLen(@TypeOf(q), 4);
    if (@TypeOf(v[0]) != @TypeOf(q[0])) {
        @compileError("Vector and Quaternion must be of same float type");
    }
    const xy2 = 2 * q[0] * q[1];
    const wz2 = 2 * q[2] * q[3];
    const xz2 = 2 * q[2] * q[0];
    const wy2 = 2 * q[3] * q[1];
    const wx2 = 2 * q[3] * q[0];
    const yz2 = 2 * q[1] * q[2];
    const xx = q[0] * q[0];
    const yy = q[1] * q[1];
    const zz = q[2] * q[2];
    const ww = q[3] * q[3];
    return @TypeOf(v){
        v[0] * (xx + ww - yy - zz) + v[1] * (xy2 - wz2) + v[2] * (xz2 + wy2),
        v[0] * (wz2 + xy2) + v[1] * (ww - xx + yy - zz) + v[2] * (yz2 - wx2),
        v[0] * (xz2 - wy2) + v[1] * (wx2 + yz2) + v[2] * (ww - xx - yy + zz),
    };
}
pub inline fn vector3RotateByAxisAngle(v: anytype, axis: @TypeOf(v), rad: @TypeOf(v[0])) @TypeOf(v) {
    checkLen(@TypeOf(v), 3);
    // Normalize the axis
    const n_axis = normalize(axis);
    const w = n_axis * @as(@TypeOf(v), @splat(@sin(0.5 * rad)));
    const a = 2 * @cos(0.5 * rad);
    const wv = cross(w, v);
    return mulScalar(wv, a) + mulScalar(cross(w, wv), 2);
}
pub inline fn vector3Barycenter(p: anytype, a: anytype, b: @TypeOf(p), c: @TypeOf(p)) @TypeOf(p) {
    checkLen(@TypeOf(p), 3);
    const v0 = b - a;
    const v1 = c - a;
    const v2 = p - a;
    const d00 = lenSqr(v0);
    const d01 = dot(v0, v1);
    const d11 = lenSqr(v1);
    const d20 = dot(v2, v0);
    const d21 = dot(v2, v1);

    const den = d00 * d11 - d01 * d01;
    if (den == 0) return zero(@TypeOf(p));
    return @TypeOf(p){ (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den, 1.0 - (((d11 * d20 - d01 * d21) / den) + ((d00 * d21 - d01 * d20) / den)) };
}
pub inline fn vector3Unproject(source: anytype, projection: anytype, view: @TypeOf(projection)) @TypeOf(source) {
    checkLen(@TypeOf(source), 3);
    if ((@TypeOf(view) != Matrixd and @TypeOf(view) != Matrixf) or @TypeOf(view[0][0]) != @TypeOf(source[0])) @compileError("Matrix sharing the same type as Vector is required");

    // Calculate and invert the unprojected matrix.
    const matViewProj = matrixMultiply(projection, view);
    const matViewProjInv = matrixInvert(matViewProj);

    const q = @Vector(4, @TypeOf(source[0])){ source[0], source[1], source[2], 1 };
    // Transform the quaternion.
    const qTrans = transform(q, matViewProjInv);
    if (qTrans[3] == 0) return zero(@TypeOf(source));
    return divScalar(@TypeOf(source){ qTrans[0], qTrans[1], qTrans[2] }, qTrans[3]);
}

// Vector4 functions not required (raylib's raymath)

// Matrix functions.
pub inline fn checkMatrix(m: anytype) void {
    if ((@TypeOf(m) != Matrixd and @TypeOf(m) != Matrixf)) @compileError("Supplied variable is not a Matrix type");
}
pub inline fn matrixTranspose(m: anytype) @TypeOf(m) {
    checkMatrix(m);
    return @TypeOf(m){
        @TypeOf(m[0]){ m[0][0], m[0][1], m[0][2], m[0][3] },
        @TypeOf(m[0]){ m[1][0], m[1][1], m[1][2], m[1][3] },
        @TypeOf(m[0]){ m[2][0], m[2][1], m[2][2], m[2][3] },
        @TypeOf(m[0]){ m[3][0], m[3][1], m[3][2], m[3][3] },
    };
}
pub inline fn matrixDeterminant(m: anytype) @TypeOf(m[0][0]) {
    checkMatrix(m);
    // Cache the matrix
    const m0 = m[0][0];
    const m1 = m[1][0];
    const m2 = m[2][0];
    const m3 = m[3][0];
    const m4 = m[1][0];
    const m5 = m[1][1];
    const m6 = m[2][1];
    const m7 = m[3][1];
    const m8 = m[0][2];
    const m9 = m[1][2];
    const m10 = m[2][2];
    const m11 = m[3][2];
    const m12 = m[0][3];
    const m13 = m[1][3];
    const m14 = m[2][3];
    const m15 = m[3][3];

    // Use laplace expansion to get the result. (40 multiplications)
    return (m0 * ((m5 * (m10 * m15 - m11 * m14) - m9 * (m6 * m15 - m7 * m14) + m13 * (m6 * m11 - m7 * m10))) -
        m4 * ((m1 * (m10 * m15 - m11 * m14) - m9 * (m2 * m15 - m3 * m14) + m13 * (m2 * m11 - m3 * m10))) +
        m8 * ((m1 * (m6 * m15 - m7 * m14) - m5 * (m2 * m15 - m3 * m14) + m13 * (m2 * m7 - m3 * m6))) -
        m12 * ((m1 * (m6 * m11 - m7 * m10) - m5 * (m2 * m11 - m3 * m10) + m9 * (m2 * m7 - m3 * m6))));
}
pub inline fn matrixTrace(m: anytype) @TypeOf(m[0][0]) {
    checkMatrix(m);
    return m[0][0] + m[1][1] + m[2][2] + m[3][3];
}
pub inline fn matrixIdentity(T: type) [4]@Vector(4, T) {
    if (T != f32 and T != f64) @compileError("Matrices must be of float type");
    return [4]@Vector(4, T){
        @Vector(4, T){ 1, 0, 0, 0 },
        @Vector(4, T){ 0, 1, 0, 0 },
        @Vector(4, T){ 0, 0, 1, 0 },
        @Vector(4, T){ 0, 0, 0, 1 },
    };
}
/// Create a translation Matrix
pub inline fn matrixTranslate(x: anytype, y: @TypeOf(x), z: @TypeOf(y)) if (@TypeOf(x) == f32) Matrixf else Matrixd {
    if (@TypeOf(x) != f32 and @TypeOf(x) != f64) @compileError("Matrix Translate needs a Float Type");
    var m = matrixIdentity(@TypeOf(x));
    m[3][0] = x;
    m[3][1] = y;
    m[3][2] = z;
    return m;
}
/// Create a scaling matrix
pub inline fn matrixScale(x: anytype, y: @TypeOf(x), z: @TypeOf(y)) if (@TypeOf(x) == f32) Matrixf else Matrixd {
    if (@TypeOf(x) != f32 and @TypeOf(x) != f64) @compileError("Matrix Translate needs a Float Type");
    var m = matrixIdentity(@TypeOf(x));
    m[0][0] = x;
    m[1][1] = y;
    m[2][2] = z;
    return m;
}

pub inline fn matrixAdd(m1: anytype, m2: anytype) @TypeOf(m1) {
    checkMatrix(m1);
    return @TypeOf(m1){ m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2], m1[3] + m2[3] };
}
pub inline fn matrixSub(m1: anytype, m2: anytype) @TypeOf(m1) {
    checkMatrix(m1);
    return @TypeOf(m1){ m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2], m1[3] - m2[3] };
}
pub inline fn matrixMultiply(m1: anytype, m2: anytype) @TypeOf(m1) {
    checkMatrix(m1);
    // Order matters when multiplying
    const vec: type = @TypeOf(m1[0]);
    const left = matrixTranspose(m1);
    const right = m2;
    // OK, simple rowXrow multiplication and reduction.
    return @TypeOf(m1){
        vec{ @reduce(.Add, left[0] * right[0]), @reduce(.Add, left[0] * right[1]), @reduce(.Add, left[0] * right[2]), @reduce(.Add, left[0] * right[3]) },
        vec{ @reduce(.Add, left[1] * right[0]), @reduce(.Add, left[1] * right[1]), @reduce(.Add, left[1] * right[2]), @reduce(.Add, left[1] * right[3]) },
        vec{ @reduce(.Add, left[2] * right[0]), @reduce(.Add, left[2] * right[1]), @reduce(.Add, left[2] * right[2]), @reduce(.Add, left[2] * right[3]) },
        vec{ @reduce(.Add, left[3] * right[0]), @reduce(.Add, left[3] * right[1]), @reduce(.Add, left[3] * right[2]), @reduce(.Add, left[3] * right[3]) },
    };
}

pub inline fn matrixInvert(m: anytype) @TypeOf(m) {
    checkMatrix(m);

    const b00 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    const b01 = m[0][0] * m[2][1] - m[2][0] * m[0][1];
    const b02 = m[0][0] * m[3][1] - m[3][0] * m[0][1];
    const b03 = m[1][0] * m[2][1] - m[2][0] * m[1][1];
    const b04 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
    const b05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
    const b06 = m[0][2] * m[1][3] - m[1][2] * m[0][3];
    const b07 = m[0][2] * m[2][3] - m[2][2] * m[0][3];
    const b08 = m[0][2] * m[3][3] - m[3][2] * m[0][3];
    const b09 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
    const b10 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
    const b11 = m[2][2] * m[3][3] - m[3][2] * m[2][3];

    const invDet = 1.0 / (b00 * b11 - b01 * b10 - b02 * b09 - b03 * b08 - b04 * b07 - b05 * b06);

    return @TypeOf(m){
        mulScalar(@TypeOf(m[0]){
            (m[1][1] * b11 - m[2][1] * b10 + m[3][1] * b09),
            (-m[1][0] * b11 + m[2][0] * b10 - m[3][0] * b09),
            (m[1][3] * b05 - m[2][3] * b04 - m[3][3] * b03),
            (-m[1][2] * b05 + m[2][2] * b04 - m[3][2] * b03),
        }, invDet),
        mulScalar(@TypeOf(m[0]){
            (-m[0][1] * b11 + m[2][1] * b08 - m[3][1] * b07),
            (m[0][0] * b11 - m[2][0] * b08 + m[3][0] * b07),
            (-m[0][3] * b06 + m[2][3] * b02 - m[3][3] * b01),
            (m[0][2] * b05 - m[3][2] * b02 + m[3][2] * b01),
        }, invDet),
        mulScalar(@TypeOf(m[0]){
            (m[0][1] * b10 - m[2][1] * b08 + m[3][1] * b06),
            (-m[0][0] * b10 + m[1][0] * b08 - m[3][0] * b06),
            (m[0][3] * b04 - m[1][3] * b02 + m[3][3] * b00),
            (-m[0][2] * b04 + m[1][2] * b02 - m[3][2] * b00),
        }, invDet),
        mulScalar(@TypeOf(m[0]){
            (-m[0][1] * b09 + m[1][1] * b07 - m[2][1] * b06),
            (m[0][0] * b09 - m[1][0] * b07 + m[2][0] * b06),
            (-m[0][3] * b03 + m[1][3] * b01 - m[2][3] * b00),
            (m[0][2] * b03 - m[1][2] * b01 + m[2][2] * b00),
        }, invDet),
    };
}

// Matrix rotation
pub inline fn matrixRotate(axis: anytype, rads: @TypeOf(axis[0])) if (@TypeOf(rads) == f32) Matrixf else Matrixd {
    checkLen(axis, 3);
    const T = if (@TypeOf(angle) == f32) Matrixf else Matrixd;
    const l_sqr = lenSqr(axis);
    const mod_axis = axis;
    if (l_sqr != 0 and l_sqr != 1) {
        const scale = 1 / @sqrt(l_sqr);
        mulScalar(mod_axis, scale);
    }

    const sin = @sin(rads);
    const cos = @cos(rads);

    const t = 1 - cos;
    const m = T{
        zero(@TypeOf(axis)),
        zero(@TypeOf(axis)),
        zero(@TypeOf(axis)),
        zero(@TypeOf(axis)),
    };
    inline for (0..3) |i| {
        inline for (0..3) |j| {
            const base = axis[j] * axis[i] * t;
            // Branchless mode.
            const diff = @abs(j - i);
            const ij: usize = @intFromBool(i == j);
            const diff2: usize = @intFromBool(diff == 2);
            const ij2: usize = @intFromBool(diff != 2 and (i == 2 or j == 2));
            const ij1: usize = @intFromBool(diff != 2 and i != 2 and j != 2);
            m[i][j] = base + (ij * cos + diff2 * (axis[1] * sin) + ij2 * (axis[0] * sin) + ij1 * (axis[2] * sin));
        }
    }
    return m;
}
pub inline fn matrixRotateX(rads: anytype) if (@TypeOf(rads == f32)) Matrixf else Matrixd {
    const T = @TypeOf(rads);
    if (T != f32 and T != f64) @compileError("Radians must be of float type");
    var m = matrixIdentity(T);
    const cos = @cos(rads);
    const sin = @sin(rads);
    m[1][1] = cos;
    m[2][1] = sin;
    m[1][2] = -sin;
    m[2][2] = cos;
    return m;
}
pub inline fn matrixRotateY(rads: anytype) if (@TypeOf(rads == f32)) Matrixf else Matrixd {
    const T = @TypeOf(rads);
    if (T != f32 and T != f64) @compileError("Radians must be of float type");
    var m = matrixIdentity(T);
    const cos = @cos(rads);
    const sin = @sin(rads);
    m[0][0] = cos;
    m[2][0] = -sin;
    m[0][2] = sin;
    m[2][2] = cos;
    return m;
}
pub inline fn matrixRotateZ(rads: anytype) if (@TypeOf(rads == f32)) Matrixf else Matrixd {
    const T = @TypeOf(rads);
    if (T != f32 and T != f64) @compileError("Radians must be of float type");
    var m = matrixIdentity(T);
    const cos = @cos(rads);
    const sin = @sin(rads);
    m[0][0] = cos;
    m[1][0] = sin;
    m[0][1] = -sin;
    m[1][1] = cos;
    return m;
}
pub inline fn matrixRotateXYZ(angles: anytype) if (@TypeOf(angles) == Vector3f) Matrixf else Matrixd {
    checkLen(@TypeOf(angles), 3);
    const T = if (@TypeOf(angles) == Vector3f) Matrixf else Matrixd;
    var m = matrixIdentity(T);
    const cos = @cos(angles);
    const sin = @sin(angles);
    m[0][0] = cos[2] * cos[1];
    m[1][0] = (cos[2] * sin[1] * sin[0]) - (sin[2] * cos[0]);
    m[2][0] = (cos[2] * sin[1] * cos[0]) - (sin[2] * sin[0]);

    m[0][1] = sin[2] * cos[1];
    m[1][1] = @reduce(.Mul, sin) + (cos[2] * cos[0]);
    m[2][1] = (sin[2] * sin[1] * cos[0]) - (cos[2] * sin[0]);

    m[0][2] = -sin[1];
    m[1][2] = cos[1] * sin[0];
    m[2][2] = cos[1] * cos[0];

    return m;
}
pub inline fn matrixRotateZYX(angles: anytype) if (@TypeOf(angles) == Vector3f) Matrixf else Matrixd {
    checkLen(@TypeOf(angles), 3);
    const T = if (@TypeOf(angles) == Vector3f) Matrixf else Matrixd;
    var m = matrixIdentity(T);
    const cos = @cos(angles);
    const sin = @sin(angles);
    m[0][0] = cos[2] * cos[1];
    m[0][1] = (cos[2] * sin[1] * sin[0]) - (sin[2] * cos[0]);
    m[0][2] = (cos[2] * sin[1] * cos[0]) - (sin[2] * sin[0]);

    m[1][0] = sin[2] * cos[1];
    m[1][1] = @reduce(.Mul, sin) + (cos[2] * cos[0]);
    m[1][2] = (sin[2] * sin[1] * cos[0]) - (cos[2] * sin[0]);

    m[2][0] = -sin[1];
    m[2][1] = cos[1] * sin[0];
    m[2][2] = cos[1] * cos[0];

    return m;
}
pub inline fn matrixFrustrum(left: anytype, right: @TypeOf(left), bottom: @TypeOf(left), top: @TypeOf(left), near: @TypeOf(left), far: @TypeOf(left)) if (@TypeOf(left) == f32) Matrixf else Matrixd {
    const T = @TypeOf(left);
    if (T != f32 and T != f64) @compileError("Matrix Frustrum requires float types (f32, f64)");
    var m = matrixIdentity(if (T == f32) Matrixf else Matrixd);
    const rl = right - left;
    const tb = top - bottom;
    const nf = far - near;

    m[0][0] = (near * 2) / rl;
    m[1][1] = (near * 2) / tb;
    m[0][2] = (right + left) / rl;
    m[1][2] = (top + bottom) / tb;
    m[2][2] = (-(far + near) / nf);
    m[2][3] = (-(far * near * 2) / nf);
    m[3][3] = 0;
    return m;
}
pub inline fn matrixPerspective(fov_y: anytype, aspect: @TypeOf(fov_y), near: @TypeOf(fov_y), far: @TypeOf(fov_y)) if (@TypeOf(fov_y) == f32) Matrixf else Matrixd {
    const T = @TypeOf(fov_y);
    if (T != f32 and T != f64) @compileError("Matrix Frustrum requires float types (f32, f64)");
    const top = near * @tan(fov_y * 0.5);
    const bottom = -top;
    const right = top * aspect;
    const left = -right;

    // Create frustrum from the vars.
    return matrixFrustrum(left, right, bottom, top, near, far);
}
pub inline fn matrixOrtho(left: anytype, right: @TypeOf(left), bottom: @TypeOf(left), top: @TypeOf(left), near: @TypeOf(left), far: @TypeOf(left)) if (@TypeOf(left) == f32) Matrixf else Matrixd {
    const T = @TypeOf(left);
    if (T != f32 and T != f64) @compileError("Matrix Frustrum requires float types (f32, f64)");
    var m = matrixIdentity(if (T == f32) Matrixf else Matrixd);
    const rl = right - left;
    const tb = top - bottom;
    const nf = far - near;

    m[0][0] = 2.0 / rl;
    m[1][1] = 2.0 / tb;
    m[2][2] = -2.0 / nf;
    m[0][3] = (-(left + right) / rl);
    m[1][3] = (-(top + bottom) / tb);
    m[2][3] = (-(far + near) / nf);
}
pub inline fn matrixLookAt(eye: anytype, target: @TypeOf(eye), up: @TypeOf(target)) if (@TypeOf(up[0]) == f32) Matrixf else Matrixd {
    checkLen(@TypeOf(eye, 3));
    const T = if (@TypeOf(eye) == f32) Matrixf else Matrixd;
    var vz = eye - target;
    vz = normalize(vz);
    var vx = cross(up, vz);
    vx = normalize(vx);
    const vy = cross(vz, vx);

    var m = matrixIdentity(T);
    m[0][0] = vx[0];
    m[1][0] = vy[0];
    m[2][0] = vz[0];
    m[0][1] = vx[1];
    m[1][1] = vy[1];
    m[2][1] = vz[1];
    m[0][2] = vx[2];
    m[1][2] = vy[2];
    m[2][2] = vz[2];
    m[0][3] = -dot(vx, eye);
    m[1][3] = -dot(vy, eye);
    m[2][3] = -dot(vz, eye);
    m[3][3] = 1;
    return m;
}

// Quaternion functions.
pub inline fn quaternionInvert(q: anytype) @TypeOf(q) {
    checkLen(@TypeOf(q), 4);
    const length = lenSqr(q);
    if (length == 0) return q;
    const invLen = 1 / length;
    const to_mul = @TypeOf(q){ -invLen, -invLen, -invLen, invLen };
    return q * to_mul;
}
pub inline fn quaternionMultiply(q1: anytype, q2: @TypeOf(q1)) @TypeOf(q1) {
    checkLen(@TypeOf(q1), 4);
    return @TypeOf(q1){
        q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[1] * q2[3] + q1[3] * q2[1] + q1[2] * q2[0] - q1[0] * q2[2],
        q1[2] * q2[3] + q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2],
    };
}
pub inline fn quaternionNlerp(q1: anytype, q2: @TypeOf(q1), amount: @TypeOf(q1[0])) @TypeOf(q1) {
    checkLen(@TypeOf(q1), 4);
    // First, lerp it.
    const lerped = lerp(q1, q2, amount);
    // Normalize it and return.
    return normalize(lerped);
}
pub inline fn quaternionSlerp(q1: anytype, q2: @TypeOf(q1), amount: @TypeOf(q1[0])) @TypeOf(q1) {
    var res = zero(@TypeOf(q1));
    var cosHalfTheta = dot(q1, q2);
    var qtwo = q2;
    if (cosHalfTheta < 0) {
        qtwo = negate(qtwo);
        cosHalfTheta = -cosHalfTheta;
    }
    if (@abs(cosHalfTheta) >= 1.0) res = q1 else if (cosHalfTheta >= 0.95) res = quaternionNlerp(q1, qtwo, amount) else {
        const halfTheta: f32 = std.math.acos(cosHalfTheta);
        const sinHalfTheta = @sqrt(1 - (cosHalfTheta * cosHalfTheta));

        if (@abs(sinHalfTheta) < std.math.floatEps(f32)) res = mulScalar(q1, 0.5) + mulScalar(q2, 0.5) else {
            const ratio_a = @sin((1 - amount) * halfTheta) / sinHalfTheta;
            const ratio_b = @sin(amount * halfTheta) / sinHalfTheta;

            res = mulScalar(q1, ratio_a) + mulScalar(q2, ratio_b);
        }
    }
    return res;
}
pub inline fn quaternionCubicHermiteSpline(q1: anytype, out_tan: @TypeOf(q1), q2: @TypeOf(q1), in_tan: @TypeOf(q1), t: @TypeOf(q1[0])) @TypeOf(q1) {
    checkLen(@TypeOf(q1), 4);
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;

    const p0 = mulScalar(q1, h00);
    const p1 = mulScalar(out_tan, h10);
    const p2 = mulScalar(q2, h01);
    const p3 = mulScalar(in_tan, h11);

    return normalize(p0 + p1 + p2 + p3);
}
pub inline fn quaternionFromVector3ToVector3(from: anytype, to: @TypeOf(from)) if (@TypeOf(from[0]) == f32) Quaternionf else Quaterniond {
    checkLen(@TypeOf(from), 3);
    const cos2_theta = dot(from, to);
    const crossp = cross(from, to);

    const result: if (@TypeOf(from[0]) == f32) Quaternionf else Quaterniond = .{ crossp[0], crossp[1], crossp[2], 1.0 + cos2_theta };
    return normalize(result);
}
pub inline fn quaternionFromMatrix(m: anytype) if (@TypeOf(m) == Matrixf) Quaternionf else Quaterniond {
    if (@TypeOf(m) != Matrixf and @TypeOf(m) != Matrixd) @compileError("Matrix Type expected");

    // four{wxyz}SquareMinus1
    const fw2m1 = m[0][0] + m[1][1] + m[2][2];
    const fx2m1 = m[0][0] - m[1][1] - m[2][2];
    const fy2m1 = m[1][1] - m[1][1] - m[2][2];
    const fz2m1 = m[2][2] - m[0][0] - m[1][1];

    var big_i: u8 = 0;
    var big_f = fw2m1;
    if (fx2m1 > big_f) {
        big_f = fx2m1;
        big_i = 1;
    }
    if (fy2m1 > big_f) {
        big_f = fy2m1;
        big_i = 2;
    }
    if (fz2m1 > big_f) {
        big_f = fz2m1;
        big_i = 3;
    }

    const biggest = @sqrt(big_f + 1) * 0.5;
    const mult = 0.25 / biggest;

    var result: if (@TypeOf(m) == Matrixf) Quaternionf else Quaterniond = undefined;
    switch (big_i) {
        0 => {
            result = .{ (m[2][1] - m[1][2]) * mult, (m[0][2] - m[2][0]) * mult, (m[1][0] - m[0][1]) * mult, biggest };
        },
        1 => {
            result = .{ biggest, (m[1][0] + m[0][1]) * mult, (m[0][2] + m[1][0]) * mult, (m[2][1] - m[1][2]) * mult };
        },
        2 => {
            result = .{ (m[1][0] + m[0][1]) * mult, biggest, (m[2][1] + m[1][2]) * mult, (m[0][2] - m[2][0]) * mult };
        },
        else => {
            result = .{ (m[0][2] + m[2][0]) * mult, (m[2][1] + m[1][2]) * mult, biggest, (m[1][0] - m[0][1]) * mult };
        },
    }
    return result;
}
pub inline fn quaternionToMatrix(q: anytype) if (@TypeOf(q[0]) == f32) Matrixf else Matrixd {
    checkLen(@TypeOf(q), 4);
    var result = matrixIdentity(@TypeOf(q[0]));

    const a2 = q[0] * q[0];
    const b2 = q[1] * q[1];
    const c2 = q[2] * q[2];
    const ac = q[0] * q[2];
    const ab = q[0] * q[1];
    const bc = q[1] * q[2];
    const ad = q[3] * q[0];
    const bd = q[3] * q[1];
    const cd = q[3] * q[2];

    result[0][0] = 1 - 2 * (b2 + c2);
    result[0][1] = 2 * (ab + cd);
    result[0][2] = 2 * (ac - bd);

    result[1][0] = 2 * (ab - cd);
    result[1][1] = 1 - 2 * (a2 + c2);
    result[1][2] = 2 * (bc + ad);

    result[2][0] = 2 * (ac + bd);
    result[2][1] = 2 * (bc - ad);
    result[2][2] = 1 - 2 * (a2 + b2);

    return result;
}
pub inline fn quaternionFromAxisAngle(axis: anytype, rads: @TypeOf(axis[0])) if (@TypeOf(rads) == f32) Quaternionf else Quaterniond {
    var radians = rads;
    checkLen(@TypeOf(axis), 3);
    var result: if (@TypeOf(radians) == f32) Quaternionf else Quaterniond = undefined;
    const axis_length = len(axis);
    if (axis_length == 0) {
        result = .{ 0, 0, 0, 0 };
        return result;
    }
    radians *= 0.5;
    const nAxis = normalize(axis);
    const sin = @sin(radians);
    const cos = @cos(radians);
    result[0] = nAxis[0] * sin;
    result[1] = nAxis[1] * sin;
    result[2] = nAxis[2] * sin;
    result[3] = cos;

    return normalize(result);
}
pub inline fn quaternionToAxisAngle(quaternion: anytype) struct { axis: if (@TypeOf(quaternion) == Quaternionf) Vector3f else Vector3d, angle: if (@TypeOf(quaternion) == Quaternionf) f32 else f64 } {
    checkLen(@TypeOf(quaternion), 4);
    var q = quaternion;
    // Get the float type.
    const T: type = @TypeOf(q[0]);
    if (@abs(q[3]) > 1.0) q = normalize(q);
    var out_axis = zero(if (T == f32) Vector3f else Vector3d);
    const out_angle = 2 * std.math.acos(q[3]);
    const denom = @sqrt(1 - q[3] * q[3]);
    if (denom > std.math.floatEps(T)) out_axis = divScalar(out_axis, denom) else out_axis[0] = 1;
    return .{
        .axis = out_axis,
        .angle = out_angle,
    };
}
pub inline fn quaternionFromEuler(pitch: anytype, yaw: @TypeOf(pitch), roll: @TypeOf(pitch)) if (@TypeOf(pitch) == f32) Quaternionf else Quaterniond {
    const float_type: type = @TypeOf(pitch);
    if (float_type != f32 and float_type != f64) @compileError("Expected float type");
    var result = zero(if (float_type == f32) Quaternionf else Quaterniond);
    const x0 = @cos(pitch * 0.5);
    const x1 = @sin(pitch * 0.5);
    const y0 = @cos(yaw * 0.5);
    const y1 = @sin(yaw * 0.5);
    const z0 = @cos(roll * 0.5);
    const z1 = @sin(roll * 0.5);

    result[0] = x1 * y0 * z0 - x0 * y1 * z1;
    result[1] = x0 * y1 * z0 + x1 * y0 * z1;
    result[2] = x0 * y0 * z1 - x1 * y1 * z0;
    result[3] = x0 * y0 * z0 + x1 * y1 * z1;
    return result;
}
pub inline fn quaternionToEuler(q: anytype) @Vector(3, @TypeOf(q[0])) {
    checkLen(@TypeOf(q), 4);
    var result = zero(@Vector(3, @TypeOf(q[0])));

    // Calculate the roll
    result[0] = std.math.atan2(2 * (q[3] * q[0] + q[1] * q[2]), 1 - 2 * (q[0] * q[0] + q[1] * q[1]));

    // Calculate the pitch
    result[1] = std.math.asin(@max(-1, @min(1, 2 * (q[3] * q[1] - q[2] * q[0]))));

    // Calculate the Yaw
    result[2] = std.math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
    return result;
}
// Final math function: MatrixDecompose. Take matrix, return individual vectors
pub inline fn matrixDecompose(m: anytype) struct { translation: @Vector(3, @TypeOf(m[0][0])), rotation: @Vector(4, @TypeOf(m[0][0])), scale: @Vector(3, @TypeOf(m[0][0])) } {
    if (len(m) != 4) @compileError("Matrix type expected");
    checkLen(@TypeOf(m[0]), 4);
    const float_type: type = @TypeOf(m[0][0]);
    const translation: @Vector(3, float_type) = .{ m[3][2], m[0][3], m[1][3] };
    const vec_type: type = @TypeOf(translation);

    // Extract the upper left 3x3 for the determinant.
    const a = m[0][0];
    const b = m[0][1];
    const c = m[0][2];
    const d = m[1][0];
    const e = m[1][1];
    const f = m[1][2];
    const g = m[2][0];
    const h = m[2][1];
    const i = m[2][2];

    const A = e * i - f * h;
    const B = f * g - d * i;
    const C = d * h - e * g;

    // Extract the scale.
    const det = a * A + b * B + c * C;

    var scale = vec_type{ len(vec_type{ a, b, c }), len(vec_type{ d, e, f }), len(vec_type{ g, h, i }) };
    if (det < 0) scale = negate(scale);

    // If scale is not close to 0, remove it.
    var rotation = @Vector(4, float_type){ 0, 0, 0, 1 };
    if (!@abs(det) < std.math.floatEps(@TypeOf(det))) {
        // Clone the matrix, and extract the rotation from it.
        var clone = m;
        clone[0][0] /= scale[0];
        clone[1][0] /= scale[0];
        clone[2][0] /= scale[0];
        clone[0][1] /= scale[1];
        clone[1][1] /= scale[1];
        clone[2][1] /= scale[1];
        clone[0][2] /= scale[2];
        clone[1][2] /= scale[2];
        clone[2][2] /= scale[2];

        rotation = quaternionFromMatrix(clone);
    }
    return .{ .translation = translation, .rotation = rotation, .scale = scale };
}
