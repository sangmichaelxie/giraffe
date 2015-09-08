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

#include "timeallocator.h"

#include <iostream>

#include <cstdint>

namespace
{

static const int32_t SUDDEN_DEATH_DIVISOR = 80;
static const double DIVISOR_MAX_RATIO = 2.0;
static const double MIN_TIME_PER_MOVE = 0.0;

// this number controls how much more time it uses in the beginning vs the end
// higher number means more time in the beginning
static const double TIME_SCALE = 2.0f;

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

		tAlloc.normalTime = (cc.GetInc() + cc.GetReading() / divisor);
		tAlloc.maxTime = cc.GetInc() + cc.GetReading() / divisor * DIVISOR_MAX_RATIO;

		if ((tAlloc.normalTime * TIME_SCALE) < (cc.GetReading() / 3.0f))
		{
			tAlloc.normalTime *= TIME_SCALE;

			if ((tAlloc.maxTime * TIME_SCALE) < (cc.GetReading() / 3.0f))
			{
				tAlloc.maxTime *= TIME_SCALE;
			}
		}

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
