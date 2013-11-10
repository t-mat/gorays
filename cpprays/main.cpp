#include <random>
#include <thread>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>
#include <map>
#include <numeric>
#include "PixelImage.hpp"
#include "ArgumentParser.hpp"
#include "Result.hpp"

#if defined(RAYS_CPP_SSE) || defined(RAYS_CPP_AVX)
typedef int vbool;

bool is_any(vbool b) {
  return 0 != b;
}

bool is_true(vbool b, int index) {
  return 0 != (b & (1 << index));
}
#endif

#if defined(RAYS_CPP_SSE) || defined(RAYS_CPP_AVX)
#include <smmintrin.h>
#include "VectorFloat32x4Sse.hpp"
#include "VectorFloat32x4x3.hpp"
#define RAYS_CPP_HAVE_FLOAT32X4
#endif

#if defined(RAYS_CPP_AVX)
#include <immintrin.h>
#include "VectorFloat32x8Avx.hpp"
#include "VectorFloat32x8x3.hpp"
#define RAYS_CPP_HAVE_FLOAT32X8
#endif

#if defined(RAYS_CPP_HAVE_FLOAT32X4)

class vector {
public:
  vector() {}
  vector(Float32x4 a) : xyzw(a) {}
  vector(float a, float b, float c) : vector(makeFloat32x4(0.0f, c, b, a)) {}

  float x() const { return get(xyzw, 0); }
  float y() const { return get(xyzw, 1); }
  float z() const { return get(xyzw, 2); }

  vector operator+(const vector r) const {
    return vector(xyzw + r.xyzw);
  }
  vector operator*(const float r) const {
    return vector(xyzw * makeFloat32x4(r));
  }
  float operator%(const vector r) const {
    return dot3(xyzw, r.xyzw);
  }
  vector operator^(vector r) const {
    return cross3(xyzw, r.xyzw);
  }
  vector operator!() const {
    return *this * (1.f / sqrtf(*this % *this));
  }

  template<class T> static T make(const vector vec) {
    return T(vec.x(), vec.y(), vec.z());
  }

  template<class T> static vector getVector(const T& t, int index) {
#if defined(_MSC_VER)
    return vector(get(t.x, index), get(t.y, index), get(t.z, index));
#else
    // FIXME (gcc 4.8.1) : calling get(Float32x[4|8], int) cause segfault
    return vector(
        reinterpret_cast<const float*>(&t.x)[index]
      , reinterpret_cast<const float*>(&t.y)[index]
      , reinterpret_cast<const float*>(&t.z)[index]
    );
#endif
  }

  Float32x4 xyzw;
};

// TODO Implement more generic way
template<class T> T make(const vector* vec);

template<> Float32x4x3 make(const vector* vec) {
   return Float32x4x3(
       makeFloat32x4(vec[3].x(), vec[2].x(), vec[1].x(), vec[0].x())
     , makeFloat32x4(vec[3].y(), vec[2].y(), vec[1].y(), vec[0].y())
     , makeFloat32x4(vec[3].z(), vec[2].z(), vec[1].z(), vec[0].z()));
  }

#if defined(RAYS_CPP_HAVE_FLOAT32X8)
template<> Float32x8x3 make(const vector* vec) {
  return Float32x8x3(
      makeFloat32x8(vec[7].x(), vec[6].x(), vec[5].x(), vec[4].x(), vec[3].x(), vec[2].x(), vec[1].x(), vec[0].x())
    , makeFloat32x8(vec[7].y(), vec[6].y(), vec[5].y(), vec[4].y(), vec[3].y(), vec[2].y(), vec[1].y(), vec[0].y())
    , makeFloat32x8(vec[7].z(), vec[6].z(), vec[5].z(), vec[4].z(), vec[3].z(), vec[2].z(), vec[1].z(), vec[0].z()));
}
#endif

#else

class vector {
public:
  vector(){}
  vector(float a, float b, float c) { _x=a; _y=b; _z=c; }

  float x() const { return _x; }
  float y() const { return _y; }
  float z() const { return _z; }

  vector operator+(vector r) const {
    return vector(_x+r._x, _y+r._y, _z+r._z);
  }
  vector operator*(float r) const {
    return vector(_x*r, _y*r, _z*r);
  }
  float operator%(vector r) const {
    return _x*r._x + _y*r._y + _z*r._z;
  }
  vector operator^(vector r) const {
    return vector(_y*r._z - _z*r._y, _z*r._x - _x*r._z, _x*r._y-_y*r._x);
  }
  vector operator!() const {
    return *this * (1.f / sqrtf(*this % *this));
  }

private:
  float _x, _y, _z;  // Vector has three float attributes.
};

#endif



typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> ClockSec;
typedef std::vector<vector> Objects;
typedef std::vector<std::string> Art;

Art readArt(std::istream& artFile) {
  Art art;
  for(std::string line; std::getline(artFile, line); ) {
    art.push_back(line);
  }
  return art;
}

Objects makeObjects(const Art& art) {
  const auto ox = 0.0f;
  const auto oy = 6.5f;
  const auto oz = -1.0f;

  Objects o;
  const auto y = oy;
  auto z = oz - static_cast<float>(art.size());
  for(const auto& line : art) {
    auto x = ox;
    for(const auto& c : line) {
      if(' ' != c) {
        o.emplace_back(x, y, z);
      }
      x += 1.0f;
    }
    z += 1.0f;
  }
  return o;
}

class Scene {
public:
  Scene(const Art& art)
    : objects(makeObjects(art))
#if defined(RAYS_CPP_HAVE_FLOAT32X4) || defined(RAYS_CPP_HAVE_FLOAT32X8)
    , objectsN(makeObjectsN(objects))
#endif
    {}

  Objects objects;

#if defined(RAYS_CPP_HAVE_FLOAT32X4) || defined(RAYS_CPP_HAVE_FLOAT32X8)

#if defined(RAYS_CPP_HAVE_FLOAT32X8)
  typedef std::vector<Float32x8x3> ObjectsN;
#elif defined(RAYS_CPP_HAVE_FLOAT32X4)
  typedef std::vector<Float32x4x3> ObjectsN;
#endif

  static ObjectsN makeObjectsN(const Objects& objectsSrc) {
    ObjectsN os;
    typedef ObjectsN::value_type V; // Float32x4x3 or Float32x8x3
    const auto nVector = V::nVector; // 4 or 8

    auto o = objectsSrc;
    while(o.size() % nVector != 0) {
      o.push_back(o.back());
    }

    for(size_t i = 0; i < o.size(); i += nVector) {
      os.emplace_back(make<V>(o.data() + i));
    }
    return os;
  }

  ObjectsN objectsN;
#endif
};

float rnd(unsigned int& seed) {
  seed += seed;
  seed ^= 1;
  if ((int)seed < 0)
    seed ^= 0x88888eef;
  return static_cast<float>(seed % 95) * (1.0f / 95.0f);
}

unsigned char clamp(float v) {
  if(v > 255.0f) {
    return 255;
  } else {
    return static_cast<unsigned char>(static_cast<int>(v));
  }
}

Pixel vectorToPixel(const vector& v) {
  Pixel p;
  p.r = clamp(v.x());
  p.g = clamp(v.y());
  p.b = clamp(v.z());
  return p;
}

enum class Status {
  kMissUpward,
  kMissDownward,
  kHit
};

struct TracerResult {
  vector n;
  Status m;
  float t;
};

TracerResult tracer(const Scene& scene, vector o, vector d) {
  auto tr = TracerResult { vector(0.0f, 0.0f, 1.0f), Status::kMissUpward, 1e9f };

  const auto p = -o.z() / d.z();

  if(.01f < p) {
    tr.t = p;
    tr.n = vector(0.0f, 0.0f, 1.0f);
    tr.m = Status::kMissDownward;
  }

#if defined(RAYS_CPP_HAVE_FLOAT32X4) || defined(RAYS_CPP_HAVE_FLOAT32X8)
  {
    int idx = -1;

    const auto& objs = scene.objectsN;
    typedef std::remove_reference<decltype(objs)>::type Vs;
    typedef Vs::value_type V; // Float32x4x3 or Float32x8x3
    const auto nVector = V::nVector; // 4 or 8
    const auto o8 = vector::make<V>(o);
    const auto d8 = vector::make<V>(d);
    for(const auto& obj : objs) {
      // There is a sphere but does the ray hits it ?
      const auto p = o8 + obj;
      const auto b = p % d8;
      const auto c = p % p - 1.0f;
      const auto b2 = b * b;
      const auto mask = compare_gt(b2, c);
      if (is_any(mask)) { // early bailout if nothing hit
        const auto q = b2 - c;
        const auto s_ = -b - sqrt(q);
        // Does the ray hit the sphere ?

        for(int j = 0; j < nVector; j++) {
          if(is_true(mask, j)) {
            const auto s = get(s_, j);
            if(s < tr.t && s > .01f) {
              const auto i = static_cast<int>(&obj - objs.data());
              idx = i * nVector + j;
              tr.t = s;
            }
          }
        }
      }
    }

    if (idx != -1) {
      const auto i = idx / nVector;
      const auto j = idx % nVector;
      const auto p = o + vector::getVector<V>(objs[i], j);
      tr.n = !(p + d * tr.t);
      tr.m = Status::kHit;
    }
  }
#else
  for (const auto& obj : scene.objects) {
    const auto p = o + obj;
    const auto b = p % d;
    const auto c = p % p - 1.0f;
    const auto b2 = b * b;

    if(b2>c) {
      const auto q = b2 - c;
      const auto s = -b - sqrtf(q);

      if(s < tr.t && s > .01f) {
        tr.t = s;
        tr.n = !(p+d*tr.t);
        tr.m = Status::kHit;
      }
    }
  }
#endif
  return tr;
}

vector sampler(const Scene& scene, vector o,vector d, unsigned int& seed) {
  //Search for an intersection ray Vs World.
  const auto tr = tracer(scene, o, d);

  if(tr.m == Status::kMissUpward) {
    const auto p = 1.f - d.z();
    return vector(1.f, 1.f, 1.f) * p;
  }

  const auto on = tr.n;
  auto h = o+d*tr.t;
  const auto l = !(vector(9.0f+rnd(seed),9.0f+rnd(seed),16.0f)+h*-1);
  auto b = l % tr.n;

  if(b < 0.0f) {
    b = 0.0f;
  } else {
    const auto tr2 = tracer(scene, h, l);
    if(tr2.m != Status::kMissUpward) {
      b = 0.0f;
    }
  }

  if(tr.m == Status::kMissDownward) {
    h = h * .2f;
    b = b * .2f + .1f;
    const auto chk = static_cast<int>(ceil(h.x()) + ceil(h.y())) & 1;
    const auto bc  = (0 != chk) ? vector(3.0f, 1.0f, 1.0f) : vector(3.0f, 3.0f, 3.0f);
    return bc * b;
  }

  const auto r = d+on*(on%d*-2.0f);               // r = The half-vector
  const auto p = pow(l % r * (b > 0.0f), 99.0f);
  return vector(p,p,p)+sampler(scene, h,r,seed)*.5f;
}

void worker(Image& image, int imageSize, const Scene& scene, unsigned int seed, int offset, int jump) {
  const auto g = !vector(-3.1f, -16.f, 1.9f);
  const auto a = !(vector(0.0f, 0.0f, 1.0f)^g) * .002f;
  const auto b = !(g^a)*.002f;
  const auto c = (a+b)*-256.0f+g;
  const auto ar = 512.0f / static_cast<float>(imageSize);
  const auto orig0 = vector(-5.0f, 16.0f, 8.0f);

  for (auto y = offset; y < imageSize; y += jump) {
    auto k = (imageSize - y - 1) * imageSize;

    for(auto x=imageSize;x--;) {
      auto p = vector(13.0f, 13.0f, 13.0f);

      for(auto r = 0; r < 64; ++r) {
        const auto t = a*((rnd(seed)-.5f)*99.0f) + b*((rnd(seed)-.5f)*99.0f);

        const auto orig = orig0 + t;
        const auto js = 16.0f;
        const auto jt = -1.0f;
        const auto ja = js * (static_cast<float>(x) * ar + rnd(seed));
        const auto jb = js * (static_cast<float>(y) * ar + rnd(seed));
        const auto jc = js;
        const auto dir = !(t*jt + a*ja + b*jb + c*jc);

        const auto s = sampler(scene, orig, dir, seed);
        p = s * 3.5f + p;
      }

      image[k++] = vectorToPixel(p);
    }
  }
}

int main(int argc, char **argv) {
  auto& outlog = std::cerr;
  const ArgumentParser cl { argc, argv, outlog }; // Don't use move constructor for GCC [4.6 , 4.8]
  const auto art = readArt(cl.artFile);
  const auto scene = Scene { art };
  auto result = Result { static_cast<size_t>(cl.times) };
  auto image = Image(cl.imageSize * cl.imageSize, Pixel::zero());

  for(auto iTimes = 0; iTimes < cl.times; ++iTimes) {
    const auto t0 = Clock::now();

    auto rgen = std::mt19937 {};
    auto threads = std::vector<std::thread>{};
    for(auto i = 0; i < cl.procs; ++i) {
      threads.emplace_back(worker, std::ref(image), cl.imageSize, std::ref(scene), rgen(), i, cl.procs);
    }
    for(auto& t : threads) {
      t.join();
    }

    const auto t1 = Clock::now();
    result.samples[iTimes] = static_cast<ClockSec>(t1 - t0).count();
    outlog << "Time taken for render " << result.samples[iTimes] << "s" << std::endl;
  }

  outlog << "Average time taken " << result.average() << "s" << std::endl;

  cl.outputFile << "P6 " << cl.imageSize << " " << cl.imageSize << " 255 "; // The PPM Header is issued
  cl.outputFile.write(reinterpret_cast<char*>(image.data()), image.size() * sizeof(image[0]));
  cl.resultFile << result.toJson();
}
