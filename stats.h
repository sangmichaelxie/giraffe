#ifndef STATS_H
#define STATS_H

#include <cstdint>

class Stat
{
public:
	Stat() : m_sum(0.0f), m_count(0) {}

	void Reset()
	{
		m_sum = 0.0f;
		m_count = 0;
	}

	void AddNumber(float x)
	{
		m_sum += x;
		++m_count;
	}

	float GetAvg()
	{
		if (m_count != 0)
		{
			return m_sum / static_cast<float>(m_count);
		}
		else
		{
			return 0.0f;
		}
	}

private:
	float m_sum;
	uint64_t m_count;
};

#endif // STATS_H
