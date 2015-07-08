#ifndef RANDOM_DEVICE_H
#define RANDOM_DEVICE_H

#include <random>
#include <mutex>
#include <thread>

#include <cstdint>

// thread-safe wrapper for std::random_device, with convenience functions
class RandomDevice
{
public:
	std::random_device::result_type operator()()
	{
		std::lock_guard<std::mutex> l(m_mutex);

		return m_rd();
	}

	std::mt19937 MakeMT()
	{
		std::lock_guard<std::mutex> l(m_mutex);

		return std::mt19937(m_rd());
	}

private:
	std::mutex m_mutex;
	std::random_device m_rd;
};

extern RandomDevice gRd;

#endif // RANDOM_DEVICE_H
