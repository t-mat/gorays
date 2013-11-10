#ifndef VECTOR_FLOAT32X4_SSE_HPP
#define VECTOR_FLOAT32X4_SSE_HPP

typedef __m128 Float32x4;

inline float get(Float32x4 v, int index) {
  float f[4];
  _mm_store_ps(f, v);
  return f[index];
}

inline Float32x4 makeFloat32x4(float f) {
  return _mm_set1_ps(f);
}

inline Float32x4 makeFloat32x4(float f0, float f1, float f2, float f3) {
  return _mm_set_ps(f0, f1, f2, f3);
}

inline Float32x4 sqrt(Float32x4 v) {
  return _mm_sqrt_ps(v);
}

inline Float32x4 rsqrt(Float32x4 v) {
  return _mm_rsqrt_ps(v);
}

inline vbool compare_gt(Float32x4 lhs, Float32x4 rhs) {
  return _mm_movemask_ps(_mm_cmp_ps(lhs, rhs, _CMP_GT_OQ));
}

#if defined(_MSC_VER)
inline Float32x4 operator+(Float32x4 lhs, Float32x4 rhs) {
  return _mm_add_ps(lhs, rhs);
}

inline Float32x4 operator-(Float32x4 lhs, Float32x4 rhs) {
  return _mm_sub_ps(lhs, rhs);
}

inline Float32x4 operator*(Float32x4 lhs, Float32x4 rhs) {
  return _mm_mul_ps(lhs, rhs);
}

inline Float32x4 operator-(Float32x4 v) {
  // Flipping sign on packed SSE floats
  // http://stackoverflow.com/questions/3361132/flipping-sign-on-packed-sse-floats
  // http://stackoverflow.com/a/3528787/2132223
  return _mm_xor_ps(v, _mm_set1_ps(-0.f));
}
#endif

inline Float32x4 operator-(Float32x4 lhs, float rhs) {
  return lhs - makeFloat32x4(rhs);
}


//
inline float dot3(Float32x4 lhs, Float32x4 rhs) {
  return _mm_cvtss_f32(_mm_dp_ps(rhs, lhs, 0x71));
}

inline Float32x4 cross3(Float32x4 lhs, Float32x4 rhs) {
  return _mm_sub_ps(
    _mm_mul_ps(
      _mm_shuffle_ps(lhs, lhs, _MM_SHUFFLE(3, 0, 2, 1)),
      _mm_shuffle_ps(rhs, rhs, _MM_SHUFFLE(3, 1, 0, 2))
    ),
    _mm_mul_ps(
      _mm_shuffle_ps(lhs, lhs, _MM_SHUFFLE(3, 1, 0, 2)),
      _mm_shuffle_ps(rhs, rhs, _MM_SHUFFLE(3, 0, 2, 1))
    )
  );
}

#endif
