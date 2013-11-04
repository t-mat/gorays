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

#if defined(RAYS_CPP_SSE) || defined(RAYS_CPP_AVX)
#include <smmintrin.h>
#endif

#if defined(RAYS_CPP_AVX)
#include <immintrin.h>
#endif

#if defined(RAYS_CPP_SSE) || defined(RAYS_CPP_AVX)

class vector {
public:
  vector() { }
  vector(__m128 a) { xyzw = a; }
  vector(float a, float b, float c) {
    xyzw = _mm_set_ps(0.0, c, b, a);
  }

  float x() const { float v[4]; _mm_store_ps(v, xyzw); return v[0]; }
  float y() const { float v[4]; _mm_store_ps(v, xyzw); return v[1]; }
  float z() const { float v[4]; _mm_store_ps(v, xyzw); return v[2]; }

  vector operator+(const vector r) const {
    return _mm_add_ps(xyzw, r.xyzw);
  }
  vector operator*(const float r) const {
    return _mm_mul_ps(_mm_set1_ps(r), xyzw);
  }
  float operator%(const vector r) const {
    float ret; _mm_store_ss(&ret, _mm_dp_ps(r.xyzw, xyzw, 0x71));
    return ret;
  }
  vector operator^(vector r) const {
    const __m128 & a = xyzw, & b = r.xyzw;
    return _mm_sub_ps(
      _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))),
      _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1)))
    );
  }
  vector operator!() const {
    return *this*(1.f/sqrtf(*this%*this));
  }

private:
    __m128 xyzw;
};

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


#if defined(RAYS_CPP_AVX)

typedef __m256 v8f;
typedef int v8bool;

v8f make_v8f(float f) {
  return _mm256_set1_ps(f);
}

v8f make_v8f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
  return _mm256_set_ps(f0, f1, f2, f3, f4, f5, f6, f7);
}

float get(v8f v, int index) {
#if defined(_MSC_VER)
  return v.m256_f32[index];
#else
  return reinterpret_cast<const float*>(&v)[index];
#endif
}

v8f sqrt(v8f v) {
  return _mm256_sqrt_ps(v);
}

v8f rsqrt(v8f v) {
  return _mm256_rsqrt_ps(v);
}

v8bool compare_gt(v8f lhs, v8f rhs) {
  return _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ));
}

bool is_any(v8bool b) {
  return 0 != b;
}

bool is_true(v8bool b, int index) {
  return 0 != (b & (1 << index));
}

#if defined(_MSC_VER)
v8f operator+(__m256 lhs, __m256 rhs) {
  return _mm256_add_ps(lhs, rhs);
}

v8f operator-(v8f lhs, v8f rhs) {
  return _mm256_sub_ps(lhs, rhs);
}

v8f operator*(v8f lhs, v8f rhs) {
  return _mm256_mul_ps(lhs, rhs);
}

v8f operator-(v8f v) {
  // Flipping sign on packed SSE floats
  // http://stackoverflow.com/questions/3361132/flipping-sign-on-packed-sse-floats
  // http://stackoverflow.com/a/3528787/2132223
  return _mm256_xor_ps(v, _mm256_set1_ps(-0.f));
}
#endif


struct vector8 {
  v8f x,y,z;  // Vector has 4 * three float attributes.
  inline vector8 operator+(vector8 r) const {return vector8(x+r.x,y+r.y,z+r.z);} //Vector add
  inline vector8 operator*(v8f r) const {return vector8(x*r,y*r,z*r);}       //Vector scaling
  inline vector8 operator*(float r) const {v8f v=make_v8f(r); return vector8(x*v,y*v,z*v);}       //Vector scaling
  inline v8f operator%(vector8 r) const {return x*r.x+y*r.y+z*r.z;}    //Vector dot product
  inline vector8 operator^(vector8 r) const {return vector8(y*r.z-z*r.y,z*r.x-x*r.z,x*r.y-y*r.x);} //Cross-product
  vector8(){}                                  //Empty constructor
  inline vector8(v8f a, v8f b,v8f c){x=a;y=b;z=c;}            //Constructor
  inline vector8(float a, float b, float c){x=make_v8f(a);y=make_v8f(b);z=make_v8f(c);}            //Constructor
  inline vector8 operator!() const {return *this*(rsqrt(*this%*this));} // Used later for normalizing the vector
  vector8(const vector* vec) {
    x = make_v8f(vec[7].x(), vec[6].x(), vec[5].x(), vec[4].x(), vec[3].x(), vec[2].x(), vec[1].x(), vec[0].x());
    y = make_v8f(vec[7].y(), vec[6].y(), vec[5].y(), vec[4].y(), vec[3].y(), vec[2].y(), vec[1].y(), vec[0].y());
    z = make_v8f(vec[7].z(), vec[6].z(), vec[5].z(), vec[4].z(), vec[3].z(), vec[2].z(), vec[1].z(), vec[0].z());
  }
  vector8(const vector vec) {
    x = make_v8f(vec.x());
    y = make_v8f(vec.y());
    z = make_v8f(vec.z());
  }
};

#endif


typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> ClockSec;
typedef std::vector<vector> Objects;
typedef std::vector<std::string> Art;
#if defined(RAYS_CPP_AVX)
typedef std::vector<vector8> Objects8;
#endif

struct Result {
  Result(size_t times)
    : samples(times, 0.0)
  {}

  std::string toJson() const {
    std::ostringstream o; // don't use auto for GCC 4.6
    o << "{\"average\":" << average() << ",";
    o << "\"samples\":[";
    for(size_t i = 0; i < samples.size(); ++i) {
      if(0 != i) {
        o << ",";
      }
      o << samples[i];
    }
    o << "]}\n";
    return o.str();
  }

  double average() const {
    return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  }

  std::vector<double> samples;
};

struct CommandLine {
  CommandLine(int argc, char* argv[])
    : megaPixels { 1.0 }
    , times { 1 }
    , procs { getMaxThreads() }
    , outputFilename { "render.ppm" }
    , resultFilename { "result.json" }
    , artFilename { "ART" }
    , home { getEnv("RAYS_HOME") }
  {
    typedef const std::string& Arg;
    typedef std::function<void(Arg)> ArgFunc;
    typedef std::map<std::string, ArgFunc> ArgFuncMap;

    const ArgFuncMap optionMap {
      { "-mp"  , [this](Arg v) { megaPixels = std::stof(v); } },
      { "-t"   , [this](Arg v) { times = std::stoi(v); } },
      { "-p"   , [this](Arg v) { procs = std::stoi(v); } },
      { "-o"   , [this](Arg v) { outputFilename = v; } },
      { "-r"   , [this](Arg v) { resultFilename = v; } },
      { "-a"   , [this](Arg v) { artFilename = v; } },
      { "-home", [this](Arg v) { home = v; } }
    };

    const auto delim = '=';
    for(auto i = 1; i < argc; ++i) {
      const auto arg = std::string { argv[i] };
      const auto pos = arg.find(delim);
      if(pos != std::string::npos) {
        const auto a = arg.substr(0, pos);
        const auto v = arg.substr(pos + 1);
        const auto it = optionMap.find(a);
        if(it != optionMap.end() && !v.empty()) {
          it->second(v);
        }
      }
    }
  }

  static std::string usage() {
    return
      "-mp=X      [        1.0] Megapixels of the rendered image\n"
      "-t=N       [          1] Times to repeat the benchmark\n"
      "-p=N       [   #Threads] Number of render threads\n"
      "-o=FILE    [render.ppm ] Output file to write the rendered image to\n"
      "-r=FILE    [result.json] Result file to write the benchmark data to\n"
      "-a=FILE    [ART        ] the art file to use for rendering\n"
      "-home=PATH [$RAYS_HOME ] RAYS folder\n";
  }

  static std::string getEnv(const std::string& env) {
    const auto* s = std::getenv(env.c_str());
    return std::string { s ? s : "" };
  }

  static int getMaxThreads(const int defaultMaxThreads = 8) {
    const auto x = std::thread::hardware_concurrency();
    return x ? x : defaultMaxThreads;
  }

  double megaPixels;
  int times;
  int procs;
  std::string outputFilename;
  std::string resultFilename;
  std::string artFilename;
  std::string home;
};

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

#if defined(RAYS_CPP_AVX)
Objects8 makeObjects8(const Objects& objectsSrc) {
  const size_t OBJ_ALIGN = 8;

  const auto objectsAlign8 = [&]() {
    Objects o = objectsSrc;
    while(o.size() % OBJ_ALIGN != 0) {
      o.push_back(o.back());
    }
    return o;
  }();

  Objects8 o;
  for(size_t i = 0; i < objectsAlign8.size(); i += OBJ_ALIGN) {
    o.push_back(vector8(objectsAlign8.data() + i));
  }
  return o;
}
#endif

class Scene {
public:
  Scene(const Art& art)
    : objects(makeObjects(art))
#if defined(RAYS_CPP_AVX)
    , objects8(makeObjects8(objects))
#endif
    {}

  Objects objects;
#if defined(RAYS_CPP_AVX)
  Objects8 objects8;
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

#if defined(RAYS_CPP_AVX)
  {
    const int nVector = 8;
    auto t = tr.t;
    auto m = (int) tr.m;
    auto n = tr.n;
    const int objects_size = static_cast<int>(scene.objects.size() + nVector - 1) / nVector;
    int idx=-1;
    float p=-o.z()/d.z();

    if(.01f<p) {
      t=p,n=vector(0,0,1),m=1;
    }

    vector8 o4 = vector8(o);
    vector8 d4(d);
    for(int i = 0; i < objects_size; i++) {
      // There is a sphere but does the ray hits it ?
      vector8 p = o4 + scene.objects8[i];
      v8f b = p % d4;
      v8f c = p % p - make_v8f(1.0f);
      v8f b2 = b*b;
      const auto mask = compare_gt(b2, c);
      if (!is_any(mask)) { // early bailout if nothing hit
        continue;
      }
      v8f q = b2 - c;
      v8f s_ = -b - sqrt(q);
      // Does the ray hit the sphere ?

      for(int j = 0; j < nVector; j++) {
        if(is_true(mask, j)) {
          float s = get(s_, j);
          if(s < t && s > .01f) {
            idx = i*nVector+j;
            t = s;
          }
        }
      }
    }

    if (idx != -1) {
      vector o2 = vector(o);
      vector p = o2 + scene.objects[idx];
      m = 2;
      n=!(p + d * t);
    }

    tr.n = n;
    tr.t = t;
    switch(m) {
      default:
      case 0: tr.m = Status::kMissUpward; break;
      case 1: tr.m = Status::kMissDownward; break;
      case 2: tr.m = Status::kHit; break;
    }
  }
#else
  const auto p = -o.z() / d.z();

  if(.01f < p) {
    tr.t = p;
    tr.n = vector(0.0f, 0.0f, 1.0f);
    tr.m = Status::kMissDownward;
  }

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

void worker(unsigned char* dst, int imageSize, const Scene& scene, unsigned int seed, int offset, int jump) {
  const auto g = !vector(-3.1f, -16.f, 1.9f);
  const auto a = !(vector(0.0f, 0.0f, 1.0f)^g) * .002f;
  const auto b = !(g^a)*.002f;
  const auto c = (a+b)*-256.0f+g;
  const auto ar = 512.0f / static_cast<float>(imageSize);
  const auto orig0 = vector(-5.0f, 16.0f, 8.0f);

  for (auto y = offset; y < imageSize; y += jump) {
    auto k = (imageSize - y - 1) * imageSize * 3;

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

      dst[k++] = clamp(p.x());
      dst[k++] = clamp(p.y());
      dst[k++] = clamp(p.z());
    }
  }
}

int main(int argc, char **argv) {
  auto& outlog = std::cerr;

  const auto cl = CommandLine { argc, argv };
  const auto artFilename = [&]() {
    std::string s;
    if(cl.artFilename == "ART" && !cl.home.empty()) {
      s += cl.home + "/";
    }
    return s + cl.artFilename;
  }();
  std::ifstream artFile { artFilename }; // don't use auto for GCC 4.6

  if (artFile.fail()) {
    outlog << "Failed to open ART file (" << artFilename << ")" << std::endl;
    std::exit(1);
  }

  const auto art = readArt(artFile);
  const auto scene = Scene { art };
  auto result = Result { static_cast<size_t>(cl.times) };

  const auto imageSize = static_cast<int>(sqrt(cl.megaPixels * 1000.0 * 1000.0));
  auto bytes = std::vector<unsigned char>(3 * imageSize * imageSize, 0);

  for(auto iTimes = 0; iTimes < cl.times; ++iTimes) {
    const auto t0 = Clock::now();

    auto rgen = std::mt19937 {};
    auto threads = std::vector<std::thread>{};
    for(auto i = 0; i < cl.procs; ++i) {
      threads.emplace_back(worker, bytes.data(), imageSize, scene, rgen(), i, cl.procs);
    }
    for(auto& t : threads) {
      t.join();
    }

    const auto t1 = Clock::now();
    result.samples[iTimes] = static_cast<ClockSec>(t1 - t0).count();
    outlog << "Time taken for render " << result.samples[iTimes] << "s" << std::endl;
  }

  outlog << "Average time taken " << result.average() << "s" << std::endl;

  std::ofstream output { cl.outputFilename }; // don't use auto for GCC 4.6
  output << "P6 " << imageSize << " " << imageSize << " 255 "; // The PPM Header is issued
  output.write(reinterpret_cast<char*>(bytes.data()), bytes.size());

  std::ofstream resultFile { cl.resultFilename }; // don't use auto for GCC 4.6
  resultFile << result.toJson();
}
