#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <immintrin.h>

typedef float v8f __attribute__ ((vector_size (32)));

//Define a vector class with constructor and operator: 'v'
struct vector {
  float x,y,z;  // Vector has three float attributes.
  inline vector operator+(vector r) const {return vector(x+r.x,y+r.y,z+r.z);} //Vector add
  inline vector operator*(float r) const {return vector(x*r,y*r,z*r);}       //Vector scaling
  inline float operator%(vector r) const {return x*r.x+y*r.y+z*r.z;}    //Vector dot product
  vector(){}                                  //Empty constructor
  inline vector operator^(vector r) const {return vector(y*r.z-z*r.y,z*r.x-x*r.z,x*r.y-y*r.x);} //Cross-product
  inline vector(float a,float b,float c){x=a;y=b;z=c;}            //Constructor
  inline vector operator!() const {return *this*(1/sqrtf(*this%*this));} // Used later for normalizing the vector
};

struct vector8 {
  v8f x,y,z;  // Vector has 4 * three float attributes.
  inline vector8 operator+(vector8 r) const {return vector8(x+r.x,y+r.y,z+r.z);} //Vector add
  inline vector8 operator*(v8f r) const {return vector8(x*r,y*r,z*r);}       //Vector scaling
  inline vector8 operator*(float r) const {v8f v=_mm256_set1_ps(r); return vector8(x*v,y*v,z*v);}       //Vector scaling
  inline v8f operator%(vector8 r) const {return x*r.x+y*r.y+z*r.z;}    //Vector dot product
  vector8(){}                                  //Empty constructor
  inline vector8 operator^(vector8 r) const {return vector8(y*r.z-z*r.y,z*r.x-x*r.z,x*r.y-y*r.x);} //Cross-product
  inline vector8(v8f a, v8f b,v8f c){x=a;y=b;z=c;}            //Constructor
  inline vector8(float a, float b, float c){x=_mm256_set1_ps(a);y=_mm256_set1_ps(b);z=_mm256_set1_ps(c);}            //Constructor
  inline vector8(vector* vec) {
    x = _mm256_set_ps(vec[7].x, vec[6].x, vec[5].x, vec[4].x, vec[3].x, vec[2].x, vec[1].x, vec[0].x);
    y = _mm256_set_ps(vec[7].y, vec[6].y, vec[5].y, vec[4].y, vec[3].y, vec[2].y, vec[1].y, vec[0].y);
    z = _mm256_set_ps(vec[7].z, vec[6].z, vec[5].z, vec[4].z, vec[3].z, vec[2].z, vec[1].z, vec[0].z);
  }
  inline vector8(vector vec) { x = _mm256_set1_ps(vec.x); y = _mm256_set1_ps(vec.y); z = _mm256_set1_ps(vec.z); }
  inline vector8 operator!() const {return *this*(_mm256_rsqrt_ps(*this%*this));} // Used later for normalizing the vector
};

const char *art[] = {
  "                   ",
  "    1111           ",
  "   1    1          ",
  "  1           11   ",
  "  1          1  1  ",
  "  1     11  1    1 ",
  "  1      1  1    1 ",
  "   1     1   1  1  ",
  "    11111     11   "
};

std::vector<vector> objects;
std::vector<vector8> objects8;

void F() {
  int nr = sizeof(art) / sizeof(char *),
  nc = strlen(art[0]);
  for (int k = nc - 1; k >= 0; k--) {
    for (int j = nr - 1; j >= 0; j--) {
      if(art[j][nc - 1 - k] != ' ') {
        objects.push_back(vector(-k, 0, -(nr - 1 - j)));
      }
    }
  }
  for (size_t i = 0; i < objects.size(); i+= 8)
    objects8.push_back(vector8(objects.data() + i));
}

float R(unsigned int& seed) {
  seed += seed;
  seed ^= 1;
  if ((int)seed < 0)
    seed ^= 0x88888eef;
  return (float)(seed % 95) / (float)95;
}

//The intersection test for line [o,v].
// Return 2 if a hit was found (and also return distance t and bouncing ray n).
// Return 0 if no hit was found but ray goes upward
// Return 1 if no hit was found but ray goes downward
int T(const vector& o,const vector& d,float& t,vector& n) {
  const int objects_size = (objects.size() + 7) / 8;
  t=1e9;
  int m=0, idx=-1;
  float p=-o.z/d.z;

  if(.01f<p)
    t=p,n=vector(0,0,1),m=1;

  vector8 o4 = vector8(o) +vector8(0,3,-4);
  vector8 d4(d);
  for(int i = 0; i < objects_size; i++) {
    // There is a sphere but does the ray hits it ?
    vector8 p = o4 + objects8[i];
    v8f b = p % d4;
    v8f c = p % p - _mm256_set1_ps(1);
    v8f b2 = b*b;
    int mask = _mm256_movemask_ps(_mm256_cmp_ps(b2, c, _CMP_GT_OQ));
    if (!mask) // early bailout if nothing hit
      continue;
    v8f q = b2 - c;
    v8f s_ = -b - _mm256_sqrt_ps(q);
    // Does the ray hit the sphere ?

    for(int j = 0; j < 8; j++) {
      if(mask & (1 << j)) {
        float s = ((float*)&s_)[j];
        if(s < t && s > .01f) {
          idx = i*8+j;
          t = s;
        }
      }
    }
  }

  if (idx != -1) {
    vector o2 = vector(o) +vector(0,3,-4);
    vector p = o2 + objects[idx];
    m = 2;
    n=!(p + d * t);
  }
  return m;
}

// (S)ample the world and return the pixel color for
// a ray passing by point o (Origin) and d (Direction)
vector S(const vector& o,const vector& d, unsigned int& seed) {
  float t;
  vector n, on;

  //Search for an intersection ray Vs World.
  int m=T(o,d,t,n);
  on = n;

  if(!m) { // m==0
    //No sphere found and the ray goes upward: Generate a sky color
    float p = 1-d.z;
    p = p*p;
    p = p*p;
    return vector(.7f,.6f,1)*p;
  }

  //A sphere was maybe hit.

  vector h=o+d*t,                    // h = intersection coordinate
  l=!(vector(9+R(seed),9+R(seed),16)+h*-1);  // 'l' = direction to light (with random delta for soft-shadows).

  //Calculated the lambertian factor
  float b=l%n;

  //Calculate illumination factor (lambertian coefficient > 0 or in shadow)?
  if(b<0||T(h,l,t,n))
    b=0;

  if(m&1) {   //m == 1
    h=h*.2f; //No sphere was hit and the ray was going downward: Generate a floor color
    return((int)(ceil(h.x)+ceil(h.y))&1?vector(3,1,1):vector(3,3,3))*(b*.2f+.1f);
  }

  vector r=d+on*(on%d*-2);               // r = The half-vector

  // Calculate the color 'p' with diffuse and specular component
  float p=l%r*(b>0);
  float p33 = p*p;
  p33 = p33*p33;
  p33 = p33*p33;
  p33 = p33*p33;
  p33 = p33*p33;
  p33 = p33*p;
  p = p33*p33*p33;

  //m == 2 A sphere was hit. Cast an ray bouncing from the sphere surface.
  return vector(p,p,p)+S(h,r,seed)*.5; //Attenuate color by 50% since it is bouncing (* .5)
}

// The main function. It generates a PPM image to stdout.
// Usage of the program is hence: ./card > erk.ppm
int main(int argc, char **argv) {
  F();

  int w = 512, h = 512;
  int num_threads = std::thread::hardware_concurrency();
  if (num_threads==0)
    //8 threads is a reasonable assumption if we don't know how many cores there are
    num_threads=8;

  if (argc > 1) {
    w = atoi(argv[1]);
  }
  if (argc > 2) {
    h = atoi(argv[2]);
  }
  if (argc > 3) {
    num_threads = atoi(argv[3]);
  }

  printf("P6 %d %d 255 ", w, h); // The PPM Header is issued

  // The '!' are for normalizing each vectors with ! operator.
  vector g=!vector(-5.5f,-16,0),       // Camera direction
    a=!(vector(0,0,1)^g)*.002f, // Camera up vector...Seem Z is pointing up :/ WTF !
    b=!(g^a)*.002f,        // The right vector, obtained via traditional cross-product
    c=(a+b)*-256+g;       // WTF ? See https://news.ycombinator.com/item?id=6425965 for more.

  int s = 3*w*h;
  char *bytes = new char[s];

  auto lambda=[&](unsigned int seed, int offset, int jump) {
    for (int y=offset; y<h; y+=jump) {    //For each row
      int k = (h - y - 1) * w * 3;

      for(int x=w;x--;) {   //For each pixel in a line
        //Reuse the vector class to store not XYZ but a RGB pixel color
        vector p(13,13,13);     // Default pixel color is almost pitch black

        //Cast 64 rays per pixel (For blur (stochastic sampling) and soft-shadows.
        for(int r=64;r--;) {
          // The delta to apply to the origin of the view (For Depth of View blur).
          vector t=a*(R(seed)-.5f)*99+b*(R(seed)-.5f)*99; // A little bit of delta up/down and left/right

          // Set the camera focal point vector(17,16,8) and Cast the ray
          // Accumulate the color returned in the p variable
          p=S(vector(17,16,8)+t, //Ray Origin
          !(t*-1+(a*(R(seed)+x)+b*(y+R(seed))+c)*16) // Ray Direction with random deltas
                                         // for stochastic sampling
          , seed)*3.5f+p; // +p for color accumulation
        }

        bytes[k++] = (char)p.x;
        bytes[k++] = (char)p.y;
        bytes[k++] = (char)p.z;
      }
    }
  };

  std::mt19937 rgen;
  std::vector<std::thread> threads;
  for(int i=0;i<num_threads;++i) {
    threads.emplace_back(lambda, rgen(), i, num_threads);
  }
  for(auto& t : threads) {
    t.join();
  }

  fwrite(bytes, 1, s, stdout);
  delete [] bytes;
}
