#ifndef MOVE_EVALUATOR_H
#define MOVE_EVALUATOR_H

#include <algorithm>
#include <iostream>
#include <limits>

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

		// used internally by ANN move evaluator
		float expectedScore;
	};

	// this struct stores things that may be useful for move ordering
	struct SearchInfo
	{
		Killer *killer = nullptr;
		TTable *tt = nullptr;
		int32_t ply = 0;
		Move hashMove = 0;
		bool isQS = false;

		// alpha and beta are from STM's perspective
		Score lowerBound = std::numeric_limits<Score>::min();
		Score upperBound = std::numeric_limits<Score>::max();
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

		NormalizeMoveInfoList(list);
	}

	virtual void PrintDiag(Board &b)
	{
		SearchInfo si;
		si.isQS = false;

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

	// implementations must override this function
	// the caller will handle normalization, but this function must handle sorting (this is to allow non-node-allocation based search order)
	// implementation can assume that list is already populated with legal moves of the correct type (QS vs non-QS)
	virtual void EvaluateMoves(Board &board, SearchInfo &si, MoveInfoList &list, MoveList &ml) = 0;
};

#endif // MOVE_EVALUATOR_H
