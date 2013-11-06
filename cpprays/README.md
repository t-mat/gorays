# cpprays

## Prerequisites

  * gcc 4.8.1/clang (currently tested with; might work with other compilers)
  * Visual Studio Express 2013 for Windows Desktop (x64)

## Features

  * The `vector` class is optimized with SSE. The optimizations gets activated if `RAYS_CPP_SSE` is defined (see below)
  * The `vector8` class is optimized with AVX. The optimizations gets activated if `RAYS_CPP_AVX` is defined (see below)

## Usage

    $ **Normal**: c++ -std=c++11 -O3 -Wall -pthread -ffast-math -mtune=native -march=native -funroll-loops -Ofast -o bin/cpprays cpprays/main.cpp
    $ **SSE**: c++ -std=c++11 -O3 -Wall -pthread -ffast-math -mtune=native -march=native -funroll-loops -Ofast -DRAYS_CPP_SSE -o bin/cpprays cpprays/main.cpp
    $ **AVX**: c++ -std=c++11 -O3 -Wall -pthread -ffast-math -mtune=native -march=native -funroll-loops -Ofast -DRAYS_CPP_AVX -o bin/cpprays cpprays/main.cpp
    $ time ./bin/cpprays > cpprays.ppm
    $ open cpprays.ppm

## Usage (Visual Studio Express 2013 for Windows Desktop)

    cd rays
    mkdir bin
    call "%VS120COMNTOOLS%..\..\VC\vcvarsall.bat" x86_amd64
    cl /nologo /EHsc /Ox /fp:fast cpprays\main.cpp /Fo:bin\cpprays.obj /Fe:bin\cpprays.exe
    bin\cpprays.exe

SSE
    cl /nologo /EHsc /Ox /fp:fast cpprays\main.cpp /Fo:bin\cpprays_sse.obj /Fe:bin\cpprays_sse.exe /DRAYS_CPP_SSE
    bin\cpprays_sse.exe

AVX
    cl /nologo /EHsc /Ox /fp:fast cpprays\main.cpp /Fo:bin\cpprays_avx.obj /Fe:bin\cpprays_avx.exe /DRAYS_CPP_AVX /arch:AVX
    bin\cpprays_avx.exe
