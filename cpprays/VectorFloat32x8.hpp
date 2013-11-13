#ifndef VECTOR_FLOAT32X8_HPP
#define VECTOR_FLOAT32X8_HPP

struct Float32x8 {
  float f[8];
};

inline float get(Float32x8 v, int index) {
  return v.f[index];
}

template<> Float32x8 make<Float32x8>(float f) {
  return Float32x8 { f, f, f, f, f, f, f, f };
}

inline Float32x8 makeFloat32x8(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
  return Float32x8 { f0, f1, f2, f3, f4, f5, f6, f7 };
}

inline Float32x8 sqrt(Float32x8 v) {
  for(int i = 0; i < 8; ++i) {
    v.f[i] = sqrt(v.f[i]);
  }
  return v;
}

inline Float32x8 rsqrt(Float32x8 v) {
  for(int i = 0; i < 8; ++i) {
    v.f[i] = 1.0f / sqrt(v.f[i]);
  }
  return v;
}

inline vbool compare_gt(Float32x8 lhs, Float32x8 rhs) {
  const vbool b = (lhs.f[0] > rhs.f[0] ? (1 << 0) : 0)
                | (lhs.f[1] > rhs.f[1] ? (1 << 1) : 0)
                | (lhs.f[2] > rhs.f[2] ? (1 << 2) : 0)
                | (lhs.f[3] > rhs.f[3] ? (1 << 3) : 0)
                | (lhs.f[4] > rhs.f[4] ? (1 << 4) : 0)
                | (lhs.f[5] > rhs.f[5] ? (1 << 5) : 0)
                | (lhs.f[6] > rhs.f[6] ? (1 << 6) : 0)
                | (lhs.f[7] > rhs.f[7] ? (1 << 7) : 0);
  return b;
}

inline Float32x8 operator+(Float32x8 lhs, Float32x8 rhs) {
  decltype(lhs) v;
  for(int i = 0; i < 8; ++i) {
    v.f[i] = lhs.f[i] + rhs.f[i];
  }
  return v;
}

inline Float32x8 operator-(Float32x8 lhs, Float32x8 rhs) {
  decltype(lhs) v;
  for(int i = 0; i < 8; ++i) {
    v.f[i] = lhs.f[i] - rhs.f[i];
  }
  return v;
}

inline Float32x8 operator*(Float32x8 lhs, Float32x8 rhs) {
  decltype(lhs) v;
  for(int i = 0; i < 8; ++i) {
    v.f[i] = lhs.f[i] * rhs.f[i];
  }
  return v;
}

inline Float32x8 operator-(Float32x8 v) {
  for(int i = 0; i < 8; ++i) {
    v.f[i] = -v.f[i];
  }
  return v;
}

inline Float32x8 operator-(Float32x8 lhs, float rhs) {
  decltype(lhs) v;
  for(int i = 0; i < 8; ++i) {
    v.f[i] = lhs.f[i] - rhs;
  }
  return v;
}

#endif
