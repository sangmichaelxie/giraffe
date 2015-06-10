CXXFLAGS_COMMON=-std=c++11 -Wall -g -I../ -o test
#CXXFLAGS=-std=c++11 -O3 -Wall -march=native -I../ -g 
CXXFLAGS=${CXXFLAGS_COMMON} -O3 -ffast-math -Wall -Wa,-q -march=native -pthread -DNDEBUG

default:
	g++-4.9 ${CXXFLAGS} -I./Eigen_dev test.cpp 

icc:
	#. /opt/intel/mkl/bin/mklvars.sh intel64
	icc -openmp -mkl=parallel ${CXXFLAGS_COMMON} -DEIGEN_USE_MKL_ALL -I./Eigen_stable -lpthread -ipo -xHost -O3 test.cpp

debug:
	g++ -std=c++11 -g -I ../ test.cpp
