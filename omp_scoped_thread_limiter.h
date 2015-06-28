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
