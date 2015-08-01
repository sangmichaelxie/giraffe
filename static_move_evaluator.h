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
#define SAMPLING

class StaticMoveEvaluator : public MoveEvaluatorIface
{
public:
#ifdef SAMPLING
	std::vector<std::string> samples;
#endif // SAMPLING

private:
	virtual void EvaluateMoves(Board &board, SearchInfo &si, MoveInfoList &list) override
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

		for (auto &mi : list)
		{
			Move mv = mi.move;

			PieceType promoType = GetPromoType(mv);

			bool isViolent = board.IsViolent(mv);

			bool isPromo = IsPromotion(mv);
			bool isQueenPromo = (promoType == WQ || promoType == BQ);
			bool isUnderPromo = (isPromo && !isQueenPromo);

			Score seeScore = SEE::StaticExchangeEvaluation(board, mv);

			SetScoreBiased(mi.move, seeScore);

			if (mv == si.hashMove)
			{
				// hash move
				mi.nodeAllocation = 3.0009f;
			}
			else if (isQueenPromo && seeScore >= 0)
			{
				// queen promos that aren't losing
				mi.nodeAllocation = 2.0008f;
			}
			else if (isViolent && seeScore >= 0 && !isUnderPromo)
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
				// killer
				mi.nodeAllocation = 1.0006f;

				for (size_t slot = 0; slot < killerMoves.GetSize(); ++slot)
				{
					if (killerMoves[slot] == mv)
					{
						// for killer moves, score is based on which slot we are in (lower = better)
						SetScoreBiased(mi.move, -static_cast<Score>(slot));
					}
				}
			}
			else if (seeScore >= 0 && !isUnderPromo)
			{
				// other non-losing moves (excluding underpomotions)
				mi.nodeAllocation = 1.0005f;
			}
			else
			{
				// losing moves and captures, and underpromos
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
					return GetScoreBiased(a.move) > GetScoreBiased(b.move);
				}
			}
		);

		for (auto &mi : list)
		{
			ClearScore(mi.move);
		}
	}
};

extern StaticMoveEvaluator gStaticMoveEvaluator;

#endif // STATIC_MOVE_EVALUATOR_H
