#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "types.h"
#include "board.h"

class EvaluatorIface
{
public:
	// return score for side to move
	virtual Score EvaluateForSTM(const Board &b, Score lowerBound, Score upperBound)
	{
		Score forWhite = EvaluateForWhite(b, lowerBound, upperBound);
		return b.GetSideToMove() == WHITE ? forWhite : -forWhite;
	}

	virtual Score EvaluateForWhite(const Board &b, Score lowerBound, Score upperBound) = 0;
};

#endif // EVALUATOR_H
