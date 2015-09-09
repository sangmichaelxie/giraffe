### Giraffe ###

Giraffe is an experimental chess engine based on temporal-difference reinforcement learning with deep neural networks. It discovers almost all its chess knowledge through self-play.

For more information, see: http://arxiv.org/abs/1509.01549

Giraffe is written in C++11.

If you decide to compile Giraffe yourself, please grab the neural network definition files (eval.net and meval.net) from the binary distribution. They must be in Giraffe's working directory when Giraffe is started. Instructions on how to generate those files will be added later.

## Gaviota Tablebases ##
To use Gaviota tablebases, set the path through the GaviotaTbPath option.

To use Gaviota tablebases with the Wb2Uci adapter, set "GaviotaTbPath=..." in Wb2Uci.eng.

## Build ##
* The Makefile contains -ltcmalloc. libtcmalloc replaces malloc/free with another implementation with thread-local caching. It is optional and doesn't really matter for playing. It can be safely removed. Or you can install the library. It's in the libgoogle-perftools-dev package on Ubuntu (and probably other Debian-based distros).
* The Makefile contains -march=native. If you want to do a compile that also runs on older CPUs, change it to something else.
* Only GCC 4.8 or later is supported for now. Intel C/C++ Compiler can be easily supported by just changing compiler options. MSVC is not supported due to use of GCC intrinsics. Patches welcomed to provide alternate code path for MSVC. Clang is not supported due to lack of OpenMP.
* Tested on Linux (GCC 4.9), OS X (GCC 4.9), Windows (MinGW-W64 GCC 5.1). GCC versions earlier than 4.8 are definitely NOT supported, due to broken regex implementation in libstdc++.