#ifndef VECTOR_FLOAT32X8X3_HPP
#define VECTOR_FLOAT32X8X3_HPP

struct Float32x8x3 {
  enum { nVector = 8 };

  Float32x8 x, y, z;  // Vector has 8 * three float attributes.

  Float32x8x3() {}                                   //Empty constructor

  Float32x8x3(Float32x8 a, Float32x8 b, Float32x8 c) //Constructor
    : x(a), y(b), z(c) {}

  Float32x8x3(float a, float b, float c)             //Constructor
    : Float32x8x3(makeFloat32x8(a), makeFloat32x8(b), makeFloat32x8(c)) {}

  Float32x8x3 operator+(Float32x8x3 r) const {           //Vector add
    return Float32x8x3(x+r.x, y+r.y, z+r.z);
  }

  Float32x8x3 operator*(Float32x8 r) const {         //Vector multiply
    return Float32x8x3(x*r, y*r, z*r);
  }

  Float32x8x3 operator*(float r) const {             //Vector scaling
    return *this * makeFloat32x8(r);
  }

  Float32x8 operator%(Float32x8x3 r) const {         //Vector dot product
    return x*r.x + y*r.y + z*r.z;
  }

  Float32x8x3 operator^(Float32x8x3 r) const {           //Cross-product
    return Float32x8x3(y*r.z-z*r.y, z*r.x-x*r.z, x*r.y-y*r.x);
  }

  Float32x8x3 operator!() const { // Used later for normalizing the vector
    return *this * (rsqrt(*this%*this));
  }
};

#endif
