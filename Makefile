CXXFLAGS_COMMON=-Wall -g -std=gnu++11 -march=sandybridge -mtune=native -I. -pthread
CXXFLAGS_RELEASE=$(CXXFLAGS_COMMON) -O3 -flto
#CXXFLAGS_PROFILE=$(CXXFLAGS_COMMON) -pg -Wno-inline -O3 -fno-inline -fno-inline-small-functions -fno-inline-functions
CXXFLAGS_PROFILE=$(CXXFLAGS_COMMON) -pg -O3 
CXXFLAGS_DEBUG=$(CXXFLAGS_COMMON)  -DDEBUG 

# this whole --whole-archive stuff is to workaround a gcc bug for cluster testing
LIBS_CLUSTER=-Wl,--whole-archive -lpthread -Wl,--no-whole-archive

all:
	$(CXX) $(CXXFLAGS_RELEASE) giraffe.cpp -o giraffe

debug:
	$(CXX) $(CXXFLAGS_DEBUG) giraffe.cpp -o giraffe

profile:
	$(CXX) $(CXXFLAGS_PROFILE) giraffe.cpp -o giraffe

cluster:
	$(CXX) -static $(CXXFLAGS_RELEASE) giraffe.cpp -o giraffe $(LIBS_CLUSTER)
