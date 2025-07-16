# zig_linalg
Zig based linear algebra library, technically a port of raymath, but using zig's SIMD Vectors. 

# Array Arithmetic
Since this uses zig's SIMD vectors, basic arithmetic works fine. 

```
const vec1 = linalg.Vector2f{0, 0};
const vec2 = linalg.Vector2f{2, 3};
const vec3 = vec1 + vec2;
```

# Caveats
At this time, only f32 and f64 are supported.
Some of the functions can easily handle ints and the like, but I just wanted to focus on floats.

# Feel free to take over. 
I mostly built this as a self teaching bit to try and learn how to think in SIMD. 
I don't really have too many intentions to maintain it beyond my own uses,
so if you want to see XYZ feature added, feel free to fork it. 