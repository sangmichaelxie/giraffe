#ifndef EVAL_H
#define EVAL_H

#include <limits>

#include <cstdint>

#include "board.h"
#include "eval_params.h"
#include "evaluator.h"

namespace Eval
{

// returns score for white
Score StaticEvaluate(const Board &b, Score lowerBound, Score upperBound);

// returns score for white
Score EvaluateMaterial(const Board &b);

class StaticEvaluator : public EvaluatorIface
{
public:
	Score EvaluateForWhiteImpl(const Board &b, Score /*lowerBound*/, Score /*upperBound*/) override
	{
		//return StaticEvaluate(b, lowerBound, upperBound);
		return EvaluateMaterial(b);
	}
} gStaticEvaluator;

}

#endif // EVAL_H
