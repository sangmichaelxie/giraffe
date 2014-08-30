CXX=g++
CXXFLAGS_COMMON=-Wall -g -Winline -std=gnu++11 -march=native
CXXFLAGS=$(CXXFLAGS_COMMON) -O3
CXXFLAGS_DEBUG=$(CXXFLAGS_COMMON)  -DDEBUG 

all:
	$(CXX) $(CXXFLAGS) giraffe.cpp -o giraffe

debug:
	$(CXX) $(CXXFLAGS_DEBUG) giraffe.cpp -o giraffe
