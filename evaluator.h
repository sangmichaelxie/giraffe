#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "types.h"
#include "board.h"

// add small offsets to prevent overflow/underflow on adding/subtracting 1 (eg. for PV search)
const static Score SCORE_MAX = std::numeric_limits<Score>::max() - 1000;
const static Score SCORE_MIN = std::numeric_limits<Score>::lowest() + 1000;

class EvaluatorIface
{
public:
	// return score for side to move
	Score EvaluateForSTM(const Board &b, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		if (b.GetSideToMove() == WHITE)
		{
			return EvaluateForWhiteImpl(b, lowerBound, upperBound);
		}
		else
		{
			return -EvaluateForWhiteImpl(b, -upperBound, -lowerBound);
		}
	}

	Score EvaluateForWhite(const Board &b, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		return EvaluateForWhiteImpl(b, lowerBound, upperBound);
	}

	// this is the only function evaluators need to implement
	virtual Score EvaluateForWhiteImpl(const Board &b, Score lowerBound, Score upperBound) = 0;
};

#endif // EVALUATOR_H
