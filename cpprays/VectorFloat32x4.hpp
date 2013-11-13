#ifndef VECTOR_FLOAT32X4_HPP
#define VECTOR_FLOAT32X4_HPP

struct Float32x4 {
  float f[4];
};

inline float get(Float32x4 v, int index) {
  return v.f[index];
}

template<> Float32x4 make<Float32x4>(float f) {
  return Float32x4 { f, f, f, f };
}

inline Float32x4 makeFloat32x4(float f0, float f1, float f2, float f3) {
  return Float32x4 { f0, f1, f2, f3 };
}

inline Float32x4 sqrt(Float32x4 v) {
  for(int i = 0; i < 4; ++i) {
    v.f[i] = sqrt(v.f[i]);
  }
  return v;
}

inline Float32x4 rsqrt(Float32x4 v) {
  for(int i = 0; i < 4; ++i) {
    v.f[i] = 1.0f / sqrt(v.f[i]);
  }
  return v;
}

inline vbool compare_gt(Float32x4 lhs, Float32x4 rhs) {
  const vbool b = (lhs.f[0] > rhs.f[0] ? (1 << 0) : 0)
                | (lhs.f[1] > rhs.f[1] ? (1 << 1) : 0)
                | (lhs.f[2] > rhs.f[2] ? (1 << 2) : 0)
                | (lhs.f[3] > rhs.f[3] ? (1 << 3) : 0);
  return b;
}

inline Float32x4 operator+(Float32x4 lhs, Float32x4 rhs) {
  decltype(lhs) v;
  for(int i = 0; i < 4; ++i) {
    v.f[i] = lhs.f[i] + rhs.f[i];
  }
  return v;
}

inline Float32x4 operator-(Float32x4 lhs, Float32x4 rhs) {
  decltype(lhs) v;
  for(int i = 0; i < 4; ++i) {
    v.f[i] = lhs.f[i] - rhs.f[i];
  }
  return v;
}

inline Float32x4 operator*(Float32x4 lhs, Float32x4 rhs) {
  decltype(lhs) v;
  for(int i = 0; i < 4; ++i) {
    v.f[i] = lhs.f[i] * rhs.f[i];
  }
  return v;
}

inline Float32x4 operator-(Float32x4 v) {
  for(int i = 0; i < 4; ++i) {
    v.f[i] = -v.f[i];
  }
  return v;
}

inline Float32x4 operator-(Float32x4 lhs, float rhs) {
  decltype(lhs) v;
  for(int i = 0; i < 4; ++i) {
    v.f[i] = lhs.f[i] - rhs;
  }
  return v;
}


//
inline float dot3(Float32x4 lhs, Float32x4 rhs) {
  return lhs.f[0]*rhs.f[0] + lhs.f[1]*rhs.f[1] + lhs.f[2]*rhs.f[2];
}

inline Float32x4 cross3(Float32x4 lhs, Float32x4 rhs) {
  return Float32x4 {
      lhs.f[1]*rhs.f[2] - lhs.f[2]*rhs.f[1]
    , lhs.f[2]*rhs.f[0] - lhs.f[0]*rhs.f[2]
    , lhs.f[0]*rhs.f[1] - lhs.f[1]*rhs.f[0]
    , 0.0f
  };
}

#endif
