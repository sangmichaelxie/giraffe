#ifndef UTIL_H
#define UTIL_H

#include <chrono>
#include <regex>
#include <sstream>

#ifdef __GNUC_MINOR__
	#ifndef __llvm__
		#define GCC_VERSION (__GNUC__ * 10000 \
							   + __GNUC_MINOR__ * 100 \
							   + __GNUC_PATCHLEVEL__)
		#if GCC_VERSION < 40900
			#error GCC 4.9.0 is the minimum required to compile this code. \
			Earlier versions have broken <regex> and WILL break this code!
		#endif
	#endif
#endif

inline double CurrentTime() { //returns current time
	return static_cast<double>(
				std::chrono::duration_cast<std::chrono::microseconds>(
					std::chrono::system_clock::now().time_since_epoch()).count()) / 1000000.0;
}

inline bool PatternMatch(std::string str, std::string pattern_str)
{
	std::regex pattern(pattern_str, std::regex_constants::extended);
	return std::regex_match(str, pattern);
}

template <typename T>
inline std::string ToStr(const T &x)
{
	std::stringstream ss;
	ss << x;
	return ss.str();
}

#endif // UTIL_H
