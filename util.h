#ifndef UTIL_H
#define UTIL_H

#include <chrono>

inline double CurrentTime() { //returns current time
	return static_cast<double>(
				std::chrono::duration_cast<std::chrono::microseconds>(
					std::chrono::system_clock::now().time_since_epoch()).count()) / 1000000.0;
}

#endif // UTIL_H
