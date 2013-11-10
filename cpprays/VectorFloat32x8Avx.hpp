#ifndef VECTOR_FLOAT32X8_AVX_HPP
#define VECTOR_FLOAT32X8_AVX_HPP

typedef __m256 Float32x8;

Float32x8 makeFloat32x8(float f) {
  return _mm256_set1_ps(f);
}

Float32x8 makeFloat32x8(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
  return _mm256_set_ps(f0, f1, f2, f3, f4, f5, f6, f7);
}

float get(Float32x8 v, int index) {
#if defined(_MSC_VER)
  return v.m256_f32[index];
#else
  return reinterpret_cast<const float*>(&v)[index];
#endif
}

Float32x8 sqrt(Float32x8 v) {
  return _mm256_sqrt_ps(v);
}

Float32x8 rsqrt(Float32x8 v) {
  return _mm256_rsqrt_ps(v);
}

vbool compare_gt(Float32x8 lhs, Float32x8 rhs) {
  return _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ));
}

#if defined(_MSC_VER)
Float32x8 operator+(Float32x8 lhs, Float32x8 rhs) {
  return _mm256_add_ps(lhs, rhs);
}

Float32x8 operator-(Float32x8 lhs, Float32x8 rhs) {
  return _mm256_sub_ps(lhs, rhs);
}

Float32x8 operator*(Float32x8 lhs, Float32x8 rhs) {
  return _mm256_mul_ps(lhs, rhs);
}

Float32x8 operator-(Float32x8 v) {
  // Flipping sign on packed SSE floats
  // http://stackoverflow.com/questions/3361132/flipping-sign-on-packed-sse-floats
  // http://stackoverflow.com/a/3528787/2132223
  return _mm256_xor_ps(v, _mm256_set1_ps(-0.f));
}
#endif

Float32x8 operator-(Float32x8 lhs, float rhs) {
  return lhs - makeFloat32x8(rhs);
}

#endif
