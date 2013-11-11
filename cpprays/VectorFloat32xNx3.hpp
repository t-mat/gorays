#ifndef VECTOR_FLOAT32XNX3_HPP
#define VECTOR_FLOAT32XNX3_HPP

template<class T> T make(float f);

template<class FLOAT32_X_N, unsigned int N>
struct Float32xNx3 {
  enum { nVector = N };
  typedef FLOAT32_X_N Float32xN;

  Float32xN x, y, z;  // Vector has N * three float attributes.

  Float32xNx3() {}                                   //Empty constructor

  Float32xNx3(Float32xN a, Float32xN b, Float32xN c) //Constructor
    : x(a), y(b), z(c) {}

  Float32xNx3(float a, float b, float c)             //Constructor
    : Float32xNx3(makeFloat32xN(a), makeFloat32xN(b), makeFloat32xN(c)) {}

  Float32xNx3 operator+(Float32xNx3 r) const {           //Vector add
    return Float32xNx3(x+r.x, y+r.y, z+r.z);
  }

  Float32xNx3 operator*(Float32xN r) const {         //Vector multiply
    return Float32xNx3(x*r, y*r, z*r);
  }

  Float32xNx3 operator*(float r) const {             //Vector scaling
    return *this * makeFloat32xN(r);
  }

  Float32xN operator%(Float32xNx3 r) const {         //Vector dot product
    return x*r.x + y*r.y + z*r.z;
  }

  Float32xNx3 operator^(Float32xNx3 r) const {           //Cross-product
    return Float32xNx3(y*r.z-z*r.y, z*r.x-x*r.z, x*r.y-y*r.x);
  }

  Float32xNx3 operator!() const { // Used later for normalizing the vector
    return *this * (rsqrt(*this%*this));
  }

  static Float32xN makeFloat32xN(float f) {
    return make<Float32xN>(f);
  }
};

#endif
