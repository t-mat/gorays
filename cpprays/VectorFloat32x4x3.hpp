#ifndef VECTOR_FLOAT32X4X3_HPP
#define VECTOR_FLOAT32X4X3_HPP

struct Float32x4x3 {
  enum { nVector = 4 };

  Float32x4 x, y, z;  // Vector has 4 * three float attributes.

  Float32x4x3() {}                                   //Empty constructor

  Float32x4x3(Float32x4 a, Float32x4 b, Float32x4 c) //Constructor
    : x(a), y(b), z(c) {}

  Float32x4x3(float a, float b, float c)             //Constructor
    : Float32x4x3(makeFloat32x4(a), makeFloat32x4(b), makeFloat32x4(c)) {}

  Float32x4x3 operator+(Float32x4x3 r) const {           //Vector add
    return Float32x4x3(x+r.x, y+r.y, z+r.z);
  }

  Float32x4x3 operator*(Float32x4 r) const {         //Vector multiply
    return Float32x4x3(x*r, y*r, z*r);
  }

  Float32x4x3 operator*(float r) const {             //Vector scaling
    return *this * makeFloat32x4(r);
  }

  Float32x4 operator%(Float32x4x3 r) const {         //Vector dot product
    return x*r.x + y*r.y + z*r.z;
  }

  Float32x4x3 operator^(Float32x4x3 r) const {           //Cross-product
    return Float32x4x3(y*r.z-z*r.y, z*r.x-x*r.z, x*r.y-y*r.x);
  }

  Float32x4x3 operator!() const { // Used later for normalizing the vector
    return *this * (rsqrt(*this%*this));
  }
};

#endif
