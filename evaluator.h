#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "types.h"
#include "board.h"

class EvaluatorIface
{
public:
	// return score for side to move
	virtual Score Evaluate(const Board &b, Score lowerBound, Score upperBound) = 0;
};

#endif // EVALUATOR_H
