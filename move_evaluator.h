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

#ifndef MOVE_EVALUATOR_H
#define MOVE_EVALUATOR_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>

#include "countermove.h"
#include "history.h"
#include "move.h"
#include "types.h"
#include "killer.h"
#include "board.h"
#include "ttable.h"

class MoveEvaluatorIface
{
public:
	struct MoveInfo
	{
		Move move;
		float nodeAllocation;

		Score seeScore;
		Score nmSeeScore;
	};

	// this struct stores things that may be useful for move ordering
	struct SearchInfo
	{
		Killer *killer = nullptr;
		TTable *tt = nullptr;
		CounterMove *counter = nullptr;
		History *history = nullptr;
		int32_t ply = 0;
		Move hashMove = 0;
		bool isQS = false;
		int64_t totalNodeBudget = 0;

		// alpha and beta are from STM's perspective
		Score lowerBound = std::numeric_limits<Score>::min();
		Score upperBound = std::numeric_limits<Score>::max();

		std::function<Score (Board &pos, Score lowerBound, Score upperBound, int64_t nodeBudget, int32_t ply)> searchFunc;
	};

	typedef FixedVector<MoveInfo, MAX_LEGAL_MOVES> MoveInfoList;

	virtual void GenerateAndEvaluateMoves(Board &board, SearchInfo &si, MoveInfoList &list)
	{
		list.Clear();

		MoveList ml;

		if (si.isQS)
		{
			board.GenerateAllLegalMoves<Board::VIOLENT>(ml);
		}
		else
		{
			board.GenerateAllLegalMoves<Board::ALL>(ml);
		}

		for (auto &move : ml)
		{
			MoveInfo mi;
			mi.move = move;
			list.PushBack(mi);
		}

		EvaluateMoves(board, si, list, ml);
	}

	virtual void PrintDiag(Board &b)
	{
		SearchInfo si;
		si.isQS = false;

		si.totalNodeBudget = 100000;

		MoveInfoList list;

		GenerateAndEvaluateMoves(b, si, list);

		for (auto &mi : list)
		{
			std::cout << b.MoveToAlg(mi.move) << ": " << mi.nodeAllocation << std::endl;
		}
	}

	void NormalizeMoveInfoList(MoveInfoList &list)
	{
		float sum = 0.0f;

		for (auto &mi : list)
		{
			sum += mi.nodeAllocation;
		}

		if (sum != 0.0f)
		{
			for (auto &mi : list)
			{
				mi.nodeAllocation /= sum;
			}
		}
	}

	// this is for search to notify the move evaluator what the actual best move turned out to be
	virtual void NotifyBestMove(Board &/*board*/, SearchInfo &/*si*/, MoveInfoList &/*list*/, Move /*bestMove*/, size_t /*movesSearched*/) {}

	// implementations must override this function
	// implementation can assume that list is already populated with legal moves of the correct type (QS vs non-QS)
	virtual void EvaluateMoves(Board &board, SearchInfo &si, MoveInfoList &list, MoveList &ml) = 0;
};

#endif // MOVE_EVALUATOR_H
