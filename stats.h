/*
	Copyright (C) 2015 Matthew Lai

	Giraffe is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	Giraffe is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
