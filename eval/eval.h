#ifndef EVAL_H
#define EVAL_H

#include <limits>

#include <cstdint>

#include "board.h"
#include "eval_params.h"
#include "evaluator.h"

// add small offsets to prevent overflow/underflow on adding/subtracting 1 (eg. for PV search)
const static Score SCORE_MAX = std::numeric_limits<Score>::max() - 1000;
const static Score SCORE_MIN = std::numeric_limits<Score>::lowest() + 1000;

namespace Eval
{

// returns score for white
Score StaticEvaluate(const Board &b, Score lowerBound, Score upperBound);

// returns score for white
Score EvaluateMaterial(const Board &b);

class StaticEvaluator : public EvaluatorIface
{
public:
	Score EvaluateForWhite(const Board &b, Score lowerBound, Score upperBound)
	{
		//return StaticEvaluate(b, lowerBound, upperBound);
		return EvaluateMaterial(b);
	}
} gStaticEvaluator;

}

#endif // EVAL_H
