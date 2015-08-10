#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "types.h"
#include "board.h"
#include "see.h"

#include <limits>

// add small offsets to prevent overflow/underflow on adding/subtracting 1 (eg. for PV search)
const static Score SCORE_MAX = std::numeric_limits<Score>::max() - 1000;
const static Score SCORE_MIN = std::numeric_limits<Score>::lowest() + 1000;

class EvaluatorIface
{
public:
	constexpr static float EvalFullScale = 10000.0f;

	// return score for side to move
	Score EvaluateForSTM(Board &b, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
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

	Score EvaluateForWhite(Board &b, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		return EvaluateForWhiteImpl(b, lowerBound, upperBound);
	}

	Score EvaluateForSTMGEE(Board &board, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		if (board.GetSideToMove() == WHITE)
		{
			return EvaluateForWhiteGEEImpl(board, lowerBound, upperBound);
		}
		else
		{
			return -EvaluateForWhiteGEEImpl(board, -upperBound, -lowerBound);
		}
	}

	Score EvaluateForWhiteGEE(Board &board, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		return EvaluateForWhiteGEEImpl(board, lowerBound, upperBound);
	}

	float UnScale(float x)
	{
		float ret = x / EvalFullScale;

		ret = std::max(ret, -1.0f);
		ret = std::min(ret, 1.0f);

		return ret;
	}

	// this is the only function evaluators need to implement
	virtual Score EvaluateForWhiteImpl(Board &b, Score lowerBound, Score upperBound) = 0;

	// evaluates the board from the perspective of the moving side by running eval on the leaf of a GEE
	// this is a generic implementation that can be overridden
	Score EvaluateForWhiteGEEImpl(Board &board, Score lowerBound, Score upperBound)
	{
		Score result = 0;

		auto staticEvalCallback = [this, &result, lowerBound, upperBound](Board &board)
		{
			result = EvaluateForWhiteImpl(board, lowerBound, upperBound);
		};

		SEE::GEERunFunc(board, staticEvalCallback);

		return result;
	}

	// this is optional
	virtual void PrintDiag(Board &/*board*/) {}
};

#endif // EVALUATOR_H
