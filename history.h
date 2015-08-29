#ifndef HISTORY_H
#define HISTORY_H

#include <vector>
#include <utility>

#include "types.h"
#include "move.h"
#include "board.h"

class History
{
public:
	History();

	void NotifyCutoff(Move move, NodeBudget nodeBudget);

	void NotifyNoCutoff(Move move, NodeBudget nodeBudget);

	// score is between 0 and 1
	float GetHistoryScore(Move move);

	void NotifyMoveMade();

private:
	// colour, from, to
	uint64_t m_cutoffCounts[2][64][64];
	uint64_t m_nonCutoffCounts[2][64][64];
};

#endif // HISTORY_H
