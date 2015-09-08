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

#ifndef STATIC_MOVE_EVALUATOR_H
#define STATIC_MOVE_EVALUATOR_H

#include <iostream>
#include <random>
#include <vector>
#include <string>

#include "move_evaluator.h"
#include "see.h"
#include "random_device.h"

// in sampling mode, we are collecting internal nodes for training
//#define SAMPLING

class StaticMoveEvaluator : public MoveEvaluatorIface
{
public:
	std::vector<std::string> samples;

	virtual void EvaluateMoves(Board &board, SearchInfo &si, MoveInfoList &list, MoveList &/*ml*/) override
	{
#ifdef SAMPLING
		static std::uniform_real_distribution<float> dist;
		static auto mt = gRd.MakeMT();

		if (dist(mt) < 0.002f)
		{
			std::string fen = board.GetFen();
			#pragma omp critical(sampleInsert)
			{
				samples.push_back(std::move(fen));
			}
		}
#endif // SAMPLING

		KillerMoveList killerMoves;

		if (si.killer)
		{
			si.killer->GetKillers(killerMoves, si.ply);
		}

		Move counterMove = 0;

		if (si.counter)
		{
			counterMove = si.counter->GetCounterMove(board);
		}

		for (auto &mi : list)
		{
			Move mv = mi.move;

			PieceType promoType = GetPromoType(mv);

			bool isViolent = board.IsViolent(mv);

			bool isPromo = IsPromotion(mv);
			bool isQueenPromo = (promoType == WQ || promoType == BQ);
			bool isUnderPromo = (isPromo && !isQueenPromo);

			mi.seeScore = SEE::StaticExchangeEvaluation(board, mv);
			mi.nmSeeScore = SEE::NMStaticExchangeEvaluation(board, mv);

			if (mv == si.hashMove)
			{
				// hash move
				mi.nodeAllocation = 3.0009f;
			}
			else if (isQueenPromo && mi.seeScore >= 0)
			{
				// queen promos that aren't losing
				mi.nodeAllocation = 2.0008f;
			}
			else if (isViolent && mi.seeScore >= 0 && !isUnderPromo)
			{
				// winning captures (excluding underpromoting captures)
				mi.nodeAllocation = 2.0007f;
			}
			else if (si.isQS)
			{
				// the above categories are the only ones we want to look at for QS
				mi.nodeAllocation = 0.0f;
			}
			else if (killerMoves.Exists(mv) && !isViolent)
			{
				bool found = false;

				// killer
				for (size_t slot = 0; slot < killerMoves.GetSize(); ++slot)
				{
					if (killerMoves[slot] == mv)
					{
						// for killer moves, score is based on which slot we are in (lower = better)
						mi.nodeAllocation = 1.100f - 0.0001f * slot;
						found = true;

						break;
					}
				}

				assert(found);
			}
			else if (mv == counterMove)
			{
				mi.nodeAllocation = 1.05f;
			}
			else if (mi.seeScore >= 0 && !isUnderPromo)
			{
				// other non-losing moves (excluding underpomotions)
				mi.nodeAllocation = 1.0000f + si.history->GetHistoryScore(mv) * 0.01f;
			}
			else if (isViolent && !isUnderPromo)
			{
				// losing captures
				mi.nodeAllocation = 0.1f;
			}
			else
			{
				// losing quiet moves and underpromos
				mi.nodeAllocation = 0.01f;
			}
		}

		std::stable_sort(list.begin(), list.end(), [](const MoveInfo &a, const MoveInfo &b)
			{
				if (a.nodeAllocation != b.nodeAllocation)
				{
					return a.nodeAllocation > b.nodeAllocation;
				}
				else
				{
					// sort based on SEE (or another source of score)
					return a.seeScore > b.seeScore;
				}
			}
		);

		NormalizeMoveInfoList(list);
	}
};

extern StaticMoveEvaluator gStaticMoveEvaluator;

#endif // STATIC_MOVE_EVALUATOR_H
