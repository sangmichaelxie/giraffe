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

#include "see.h"
#include "eval/eval_params.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include <cstdint>

namespace SEE
{

// best tactical result for the moving side
Score StaticExchangeEvaluation(Board &board, Move mv)
{
	board.ResetSee();

	// convert the move to SEE format
	PieceType pt;
	Square from;
	Square to;

	pt = GetPieceType(mv);
	from = GetFromSquare(mv);
	to = GetToSquare(mv);

	PieceType capturedPT = board.ApplyMoveSee(pt, from, to);

	// the first move is forced
	Score ret = 0;

	if (capturedPT != EMPTY)
	{
		ret = SEE_MAT[capturedPT] - StaticExchangeEvaluationSq(board, to);
	}
	else
	{
		ret = -StaticExchangeEvaluationSq(board, to);
	}

	board.UndoMoveSee();

	return ret;
}

Score SEEMap(Board &board, Square sq)
{
	board.ResetSee();

	return -StaticExchangeEvaluationSq(board, sq, true);
}

Score StaticExchangeEvaluationSq(Board &board, Square sq, bool forced)
{
	Score ret = 0;

	PieceType pt;
	Square from;

	bool hasMoreCapture = board.GenerateSmallestCaptureSee(pt, from, sq);

	if (hasMoreCapture)
	{
		PieceType capturedPT = board.ApplyMoveSee(pt, from, sq);

		if (forced)
		{
			// in forced mode, we are trying to build a SEE map, so we assume the square to be empty
			// (even if it's not)
			ret = -StaticExchangeEvaluationSq(board, sq);
		}
		else
		{
			ret = std::max(0, SEE_MAT[capturedPT] - StaticExchangeEvaluationSq(board, sq));
		}

		board.UndoMoveSee();
	}
	else
	{
		if (forced)
		{
			// if the move is forced and we don't have a move, return worst result
			ret = -SEE_MAT[WK];
		}
	}

	return ret;
}

Score NMStaticExchangeEvaluation(Board &board, Move mv)
{
	Score ret = 0;

	if (!board.InCheck())
	{
		board.MakeNullMove();

		board.ResetSee();

		// positive value means we should move this piece (opponent can win it otherwise)
		ret = StaticExchangeEvaluationSq(board, GetFromSquare(mv));

		board.UndoMove();
	}

	return ret;
}

Score GlobalExchangeEvaluation(Board &board, std::vector<Move> &pv, Score currentEval, Score lowerBound, Score upperBound)
{
	assert(pv.empty());

	// try standpat
	if (currentEval >= upperBound)
	{
		return currentEval;
	}
	else if (currentEval > lowerBound)
	{
		lowerBound = currentEval;
	}

	MoveList captures;
	board.GenerateAllLegalMoves<Board::VIOLENT>(captures);

	std::vector<Move> subPv;

	for (size_t i = 0; i < captures.GetSize(); ++i)
	{
		Score see = StaticExchangeEvaluation(board, captures[i]);

		// we only want to search positive SEEs (not even neutral ones), and only if it can possibly improve lowerBound
		if ((see < 0) || ((currentEval + see) <= lowerBound))
		{
			continue;
		}

		subPv.clear();

		PieceType capturedPt = board.GetCapturedPieceType(captures[i]);

		board.ApplyMove(captures[i]);

		Score score = -GlobalExchangeEvaluation(board, subPv, -(currentEval + SEE_MAT[capturedPt]), -upperBound, -lowerBound);

		board.UndoMove();

		if (score >= upperBound)
		{
			return score;
		}

		if (score > lowerBound)
		{
			lowerBound = score;

			pv.clear();
			pv.push_back(captures[i]);
			pv.insert(pv.end(), subPv.begin(), subPv.end());
		}
	}

	return lowerBound;
}

void GEERunFunc(Board &board, std::function<void(Board &b)> func)
{
	std::vector<Move> pv;

	GlobalExchangeEvaluation(board, pv);

	board.ApplyVariation(pv);

	func(board);

	for (size_t i = 0; i < pv.size(); ++i)
	{
		board.UndoMove();
	}
}

bool RunSeeTest(std::string fen, std::string move, Score expectedScore)
{
	std::cout << "Checking SEE for " << fen << ", <= " << move << std::endl;

	Board b(fen);
	Move mv = b.ParseMove(move);

	if (mv == 0)
	{
		std::cerr << "Failed to parse move " << move << std::endl;
		return false;
	}

	Score see = StaticExchangeEvaluation(b, mv);

	if (see != expectedScore)
	{
		std::cerr << "Expected: " << expectedScore << " Got: " << see << std::endl;
		return false;
	}

	b.CheckBoardConsistency();

	std::cout << "Passed" << std::endl;

	return true;
}

void DebugRunSeeTests()
{
	// basic white capture, Rxd5
	if (!RunSeeTest("7k/8/8/3p4/8/3R4/8/K7 w - - 0 1", "d3d5", 100)) { abort(); }

	// basic black capture, exf5
	if (!RunSeeTest("7k/8/8/4p3/5R2/8/8/K7 b - - 0 1", "e5f4", 600)) { abort(); }

	// simple exchange, exf4 Rxf4
	if (!RunSeeTest("6k1/8/8/4p3/5R1R/8/8/K7 b - - 0 1", "e5f4", 500)) { abort(); }

	// decide to not capture, exf4
	if (!RunSeeTest("7k/8/8/4p3/5R2/8/8/K7 b - - 0 1", "e5f4", 600)) { abort(); }

	// decide to not recapture due to discovered attacker, Rxe6
	if (!RunSeeTest("7k/4q3/4q3/8/4R3/4R3/8/K7 w - - 0 1", "e4e6", 1200)) { abort(); }

	// recapture without the discovered attacker, Rxe6 Qxe6
	if (!RunSeeTest("7k/4q3/4q3/8/4R3/8/8/K7 w - - 0 1", "e4e6", 600)) { abort(); }

	// complex capture sequence, cxd4 exd4 Nxd4
	if (!RunSeeTest("4q2k/3q2b1/8/2p5/3P4/4P3/3Rn3/K2R4 b - - 0 1", "c5d4", 100)) { abort(); }

	// similar situation, but less defender
	if (!RunSeeTest("4q2k/3q4/8/2p5/3P4/4P3/3R4/K2R4 b - - 0 1", "c5d4", 0)) { abort(); }

	// queen defender blocked by pawn, cxd4
	if (!RunSeeTest("7k/q7/8/2p5/3P4/8/3R4/6K1 b - - 0 1", "c5d4", 100)) { abort(); }

	// bad capture by black, Nxd4 Rxd4
	if (!RunSeeTest("7k/q7/2n5/8/3P4/8/3R4/3R2K1 b - - 0 1", "c6d4", -300)) { abort(); }

	// bad capture by white, Rxd4 Nxd4
	if (!RunSeeTest("7k/q7/2n5/8/3p4/8/3R4/3R2K1 w - - 0 1", "d2d4", -500)) { abort(); }

	// white non-capture, losing
	if (!RunSeeTest("2r4k/1P6/8/4q1nr/7p/5N2/K7/8 w - - 0 1", "f3e1", -400)) { abort(); }

	// white non-capture, non-losing
	if (!RunSeeTest("2r4k/1P6/8/4q1nr/7p/5N2/K7/8 w - - 0 1", "f3d2", 0)) { abort(); }
}

}
