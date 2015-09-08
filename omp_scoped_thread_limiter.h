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

#ifndef OMP_SCOPED_THREAD_LIMITER_H
#define OMP_SCOPED_THREAD_LIMITER_H

#include <omp.h>

class ScopedThreadLimiter
{
public:
	ScopedThreadLimiter(int limit)
	{
		m_originalLimit = omp_get_max_threads();

		if (limit < m_originalLimit)
		{
			omp_set_num_threads(limit);
		}
	}

	~ScopedThreadLimiter()
	{
		omp_set_num_threads(m_originalLimit);
	}

	ScopedThreadLimiter &operator=(const ScopedThreadLimiter &other) = delete;
	ScopedThreadLimiter(const ScopedThreadLimiter &other) = delete;

private:
	int m_originalLimit;
};

#endif // OMP_SCOPED_THREAD_LIMITER_H
