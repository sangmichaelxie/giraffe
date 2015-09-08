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
	virtual Score EvaluateForSTM(Board &b, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
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

	virtual Score EvaluateForWhite(Board &b, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		return EvaluateForWhiteImpl(b, lowerBound, upperBound);
	}

	virtual Score EvaluateForSTMGEE(Board &board, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
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

	virtual Score EvaluateForWhiteGEE(Board &board, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		return EvaluateForWhiteGEEImpl(board, lowerBound, upperBound);
	}

	virtual void BatchEvaluateForSTMGEE(std::vector<Board> &positions, std::vector<Score> &results, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		// check that they all have the same stm
		Color stm = positions[0].GetSideToMove();

		for (size_t i = 1; i < positions.size(); ++i)
		{
			assert(positions[i].GetSideToMove() == stm);
		}

		if (stm == WHITE)
		{
			BatchEvaluateForWhiteGEEImpl(positions, results, lowerBound, upperBound);
		}
		else
		{
			BatchEvaluateForWhiteGEEImpl(positions, results, -upperBound, -lowerBound);

			for (auto &x : results)
			{
				x *= -1;
			}
		}
	}

	virtual void BatchEvaluateForWhiteGEE(std::vector<Board> &positions, std::vector<Score> &results, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		BatchEvaluateForWhiteGEEImpl(positions, results, lowerBound, upperBound);
	}

	virtual float UnScale(float x)
	{
		float ret = x / EvalFullScale;

		ret = std::max(ret, -1.0f);
		ret = std::min(ret, 1.0f);

		return ret;
	}

	// this is the only function evaluators need to implement
	virtual Score EvaluateForWhiteImpl(Board &b, Score lowerBound, Score upperBound) = 0;

	// this allows evaluators to evaluate multiple positions at once
	// default implementation does it one at a time
	virtual void BatchEvaluateForWhiteImpl(std::vector<Board> &positions, std::vector<Score> &results, Score lowerBound, Score upperBound)
	{
		results.resize(positions.size());

		for (size_t i = 0; i < positions.size(); ++i)
		{
			results[i] = EvaluateForWhiteImpl(positions[i], lowerBound, upperBound);
		}
	}

	// evaluates the board from the perspective of the moving side by running eval on the leaf of a GEE
	// this is a generic implementation that can be overridden
	virtual Score EvaluateForWhiteGEEImpl(Board &board, Score lowerBound, Score upperBound)
	{
		Score result = 0;

		auto staticEvalCallback = [this, &result, lowerBound, upperBound](Board &board)
		{
			result = EvaluateForWhiteImpl(board, lowerBound, upperBound);
		};

		SEE::GEERunFunc(board, staticEvalCallback);

		return result;
	}

	virtual void BatchEvaluateForWhiteGEEImpl(std::vector<Board> &positions, std::vector<Score> &results, Score lowerBound, Score upperBound)
	{
		std::vector<Board> leafPositions;

		leafPositions.reserve(positions.size());

		auto vectorInsertCallback = [this, &leafPositions](Board &board)
		{
			leafPositions.push_back(board);
		};

		for (size_t i = 0; i < positions.size(); ++i)
		{
			SEE::GEERunFunc(positions[i], vectorInsertCallback);
		}

		BatchEvaluateForWhiteImpl(leafPositions, results, lowerBound, upperBound);
	}

	// this is optional
	virtual void PrintDiag(Board &/*board*/) {}
};

#endif // EVALUATOR_H
