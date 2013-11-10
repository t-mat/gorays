#ifndef PIXELIMAGE_H
#define PIXELIMAGE_H

struct Pixel {
  uint8_t r, g, b;

  static Pixel zero() {
    return Pixel { 0, 0, 0 };
  }
};

typedef std::vector<Pixel> Image;

#endif
