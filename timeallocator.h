#ifndef TIMEALLOCATOR_H
#define TIMEALLOCATOR_H

#include "chessclock.h"
#include "search.h"

// allocate time for a move given the engine's current clock
Search::TimeAllocation AllocateTime(const ChessClock &cc);

#endif // TIMEALLOCATOR_H
