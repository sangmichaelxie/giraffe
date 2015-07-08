CXX=g++-4.9
CXXFLAGS_COMMON=-Wall -Wextra -g -std=gnu++11 -mtune=native -Wa,-q -ffast-math -I. -IEigen_dev -pthread -fopenmp
CXXFLAGS_RELEASE=$(CXXFLAGS_COMMON) -O3 -L. -ltcmalloc
CXXFLAGS_PROFILE=$(CXXFLAGS_COMMON) -pg -Os
CXXFLAGS_DEBUG=$(CXXFLAGS_COMMON)  -DDEBUG 
CXXFLAGS_FAST_COMPILE=$(CXXFLAGS_COMMON) -O0

# this whole --whole-archive stuff is to workaround a gcc bug for cluster testing
LIBS_CLUSTER=-Wl,--whole-archive -lpthread -Wl,--no-whole-archive

all:
	$(CXX) $(CXXFLAGS_RELEASE) -march=native giraffe.cpp -o giraffe

debug:
	$(CXX) $(CXXFLAGS_DEBUG) giraffe.cpp -o giraffe

profile:
	$(CXX) $(CXXFLAGS_PROFILE) giraffe.cpp -o giraffe

cluster:
	$(CXX) -static -march=sandybridge $(CXXFLAGS_RELEASE) giraffe.cpp -o giraffe $(LIBS_CLUSTER)

fast_compile:
	$(CXX) $(CXXFLAGS_FAST_COMPILE) giraffe.cpp -o giraffe

gpu_osx:
	$(CXX) $(CXXFLAGS_RELEASE) -march=native -framework OpenCL -DVIENNACL_WITH_OPENCL giraffe.cpp -o giraffe

gpu:
	$(CXX) $(CXXFLAGS_RELEASE) -march=native -lOpenCL -DVIENNACL_WITH_OPENCL giraffe.cpp -o giraffe
	
windows:
	g++ $(CXXFLAGS_COMMON) -O3 giraffe.cpp -o giraffe
	strip -g -s giraffe.exe
