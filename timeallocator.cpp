#include "timeallocator.h"

#include <iostream>

#include <cstdint>

namespace
{

static const int32_t SUDDEN_DEATH_DIVISOR = 50;
static const double DIVISOR_MAX_RATIO = 2.0;
static const double MIN_TIME_PER_MOVE = 0.005;

}

Search::TimeAllocation AllocateTime(const ChessClock &cc)
{
	Search::TimeAllocation tAlloc;

	if (cc.GetMode() == ChessClock::EXACT_MODE)
	{
		tAlloc.normalTime = cc.GetInc();
		tAlloc.maxTime = cc.GetInc();
	}
	else if (cc.GetMode() == ChessClock::CONVENTIONAL_INCREMENTAL_MODE)
	{
		int32_t divisor = cc.GetMovesUntilNextPeriod();

		if (divisor == 0)
		{
			// sudden death mode
			divisor = SUDDEN_DEATH_DIVISOR;
		}

		tAlloc.normalTime = cc.GetInc() + cc.GetReading() / divisor;
		tAlloc.maxTime = cc.GetInc() + cc.GetReading() / divisor * DIVISOR_MAX_RATIO;

		tAlloc.normalTime = std::max(tAlloc.normalTime, MIN_TIME_PER_MOVE);
		tAlloc.maxTime = std::max(tAlloc.maxTime, MIN_TIME_PER_MOVE);

		std::cout << "# Allocated " << tAlloc.normalTime << ", " << tAlloc.maxTime << " seconds" << std::endl;
	}
	else
	{
		std::cout << "Error (unknown time control mode)" << std::endl;
	}

	return tAlloc;
}
