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
		if (b.GetSideToMove() == WHITE)
		{
			return EvaluateForWhite(b, lowerBound, upperBound);
		}
		else
		{
			return -EvaluateForWhite(b, -upperBound, -lowerBound);
		}
	}

	virtual Score EvaluateForWhite(const Board &b, Score lowerBound, Score upperBound) = 0;
};

#endif // EVALUATOR_H
