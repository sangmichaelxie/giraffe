CXX=g++
CXXFLAGS=-Wall -Winline -g -march=native -O3

all:
	$(CXX) $(CXXFLAGS) giraffe.cpp -o giraffe
