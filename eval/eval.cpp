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

#include "eval.h"
#include "types.h"
#include "bit_ops.h"
#include "magic_moves.h"
#include "evaluator.h"

#include <cmath>

namespace Eval
{

Score ScalePhase(Score openingScore, Score endgameScore, Phase phase)
{
	Score diff = openingScore - endgameScore;

	return endgameScore + diff * phase / MAX_PHASE;
}

template <Color COLOR>
Score EvaluatePawns(uint64_t bb, Phase phase, uint64_t &pawnAttacksOut)
{
	Score ret = 0;

	while (bb)
	{
		uint32_t idx = Extract(bb);

		if (COLOR == WHITE)
		{
			pawnAttacksOut |= PAWN_ATK[idx][0];
		}
		else
		{
			pawnAttacksOut |= PAWN_ATK[idx][1];
		}

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(PAWN_PCSQ[idx] * PAWN_PCSQ_MULTIPLIERS[0], PAWN_PCSQ[idx] * PAWN_PCSQ_MULTIPLIERS[1], phase);
	}

	return ret;
}

template <Color COLOR>
Score EvaluateKnights(uint64_t bb, Phase phase, uint64_t safeDestinations)
{
	Score ret = 0;

	while (bb)
	{
		uint32_t idx = Extract(bb);

		uint32_t mobility = PopCount(KNIGHT_ATK[idx] & safeDestinations);

		ret += ScalePhase(KNIGHT_MOBILITY[0][mobility] * MOBILITY_MULTIPLIERS[0],
						  KNIGHT_MOBILITY[1][mobility] * MOBILITY_MULTIPLIERS[1], phase);

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(KNIGHT_PCSQ[0][idx], KNIGHT_PCSQ[1][idx], phase);
	}

	return ret;
}

template <Color COLOR>
Score EvaluateBishops(uint64_t bb, Phase phase, uint64_t safeDestinations, uint64_t occupancy)
{
	Score ret = 0;

	if (PopCount(bb) >= 2)
	{
		ret += ScalePhase(BISHOP_PAIR_BONUS[0], BISHOP_PAIR_BONUS[1], phase);
	}

	while (bb)
	{
		uint32_t idx = Extract(bb);

		uint32_t mobility = PopCount(Bmagic(idx, occupancy) & safeDestinations);

		ret += ScalePhase(BISHOP_MOBILITY[0][mobility] * MOBILITY_MULTIPLIERS[0],
						  BISHOP_MOBILITY[1][mobility] * MOBILITY_MULTIPLIERS[1], phase);

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(BISHOP_PCSQ[0][idx], BISHOP_PCSQ[1][idx], phase);
	}

	return ret;
}

template <Color COLOR>
Score EvaluateRooks(uint64_t bb, Phase phase, uint64_t safeDestinations, uint64_t occupancy)
{
	Score ret = 0;

	while (bb)
	{
		uint32_t idx = Extract(bb);

		uint32_t mobility = PopCount(Rmagic(idx, occupancy) & safeDestinations);

		ret += ScalePhase(ROOK_MOBILITY[0][mobility] * MOBILITY_MULTIPLIERS[0],
						  ROOK_MOBILITY[1][mobility] * MOBILITY_MULTIPLIERS[1], phase);

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(ROOK_PCSQ[0][idx], ROOK_PCSQ[1][idx], phase);
	}

	return ret;
}

template <Color COLOR>
Score EvaluateQueens(uint64_t bb, Phase phase, uint64_t safeDestinations, uint64_t occupancy)
{
	Score ret = 0;

	while (bb)
	{
		uint32_t idx = Extract(bb);

		uint32_t mobility = PopCount(Qmagic(idx, occupancy) & safeDestinations);

		ret += ScalePhase(QUEEN_MOBILITY[0][mobility] * MOBILITY_MULTIPLIERS[0],
						  QUEEN_MOBILITY[1][mobility] * MOBILITY_MULTIPLIERS[1], phase);

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(QUEEN_PCSQ[0][idx], QUEEN_PCSQ[1][idx], phase);
	}

	return ret;
}

template <Color COLOR>
Score EvaluateKings(uint64_t bb, Phase phase)
{
	Score ret = 0;

	while (bb)
	{
		uint32_t idx = Extract(bb);

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(KING_PCSQ[0][idx], KING_PCSQ[1][idx], phase);
	}

	return ret;
}

Score StaticEvaluate(const Board &b, Score /*lowerBound*/, Score /*upperBound*/)
{
	Score ret = 0;

	uint32_t WQCount = PopCount(b.GetPieceTypeBitboard(WQ));
	uint32_t WRCount = PopCount(b.GetPieceTypeBitboard(WR));
	uint32_t WBCount = PopCount(b.GetPieceTypeBitboard(WB));
	uint32_t WNCount = PopCount(b.GetPieceTypeBitboard(WN));
	uint32_t WPCount = PopCount(b.GetPieceTypeBitboard(WP));

	uint32_t BQCount = PopCount(b.GetPieceTypeBitboard(BQ));
	uint32_t BRCount = PopCount(b.GetPieceTypeBitboard(BR));
	uint32_t BBCount = PopCount(b.GetPieceTypeBitboard(BB));
	uint32_t BNCount = PopCount(b.GetPieceTypeBitboard(BN));
	uint32_t BPCount = PopCount(b.GetPieceTypeBitboard(BP));

	Phase phase =
		WQCount * Q_PHASE_CONTRIBUTION +
		BQCount * Q_PHASE_CONTRIBUTION +
		WRCount * R_PHASE_CONTRIBUTION +
		BRCount * R_PHASE_CONTRIBUTION +
		WBCount * B_PHASE_CONTRIBUTION +
		BBCount * B_PHASE_CONTRIBUTION +
		WNCount * N_PHASE_CONTRIBUTION +
		BNCount * N_PHASE_CONTRIBUTION +
		WPCount * P_PHASE_CONTRIBUTION +
		BPCount * P_PHASE_CONTRIBUTION;

	if (phase > MAX_PHASE)
	{
		// this can happen on custom positions and in case of promotions
		phase = MAX_PHASE;
	}

	ret += WQCount * ScalePhase(MAT[0][WQ], MAT[1][WQ], phase);
	ret += WRCount * ScalePhase(MAT[0][WR], MAT[1][WR], phase);
	ret += WBCount * ScalePhase(MAT[0][WB], MAT[1][WB], phase);
	ret += WNCount * ScalePhase(MAT[0][WN], MAT[1][WN], phase);
	ret += WPCount * ScalePhase(MAT[0][WP], MAT[1][WP], phase);

	ret -= BQCount * ScalePhase(MAT[0][WQ], MAT[1][WQ], phase);
	ret -= BRCount * ScalePhase(MAT[0][WR], MAT[1][WR], phase);
	ret -= BBCount * ScalePhase(MAT[0][WB], MAT[1][WB], phase);
	ret -= BNCount * ScalePhase(MAT[0][WN], MAT[1][WN], phase);
	ret -= BPCount * ScalePhase(MAT[0][WP], MAT[1][WP], phase);

	uint64_t occupancy = b.GetOccupiedBitboard<WHITE>() | b.GetOccupiedBitboard<BLACK>();

	uint64_t attackedByWhitePawns = 0ULL;
	uint64_t attackedByBlackPawns = 0ULL;

	ret += EvaluatePawns<WHITE>(b.GetPieceTypeBitboard(WP), phase, attackedByWhitePawns);
	ret -= EvaluatePawns<BLACK>(b.GetPieceTypeBitboard(BP), phase, attackedByBlackPawns);

	// safe destinations are empty squares or squares with enemy pieces
	// these squares must not be defended by enemy pawns
	uint64_t whiteSafeDestinations = ~b.GetOccupiedBitboard<WHITE>() & ~attackedByBlackPawns;
	uint64_t blackSafeDestinations = ~b.GetOccupiedBitboard<BLACK>() & ~attackedByWhitePawns;

	ret += EvaluateKnights<WHITE>(b.GetPieceTypeBitboard(WN), phase, whiteSafeDestinations);
	ret -= EvaluateKnights<BLACK>(b.GetPieceTypeBitboard(BN), phase, blackSafeDestinations);

	ret += EvaluateBishops<WHITE>(b.GetPieceTypeBitboard(WB), phase, whiteSafeDestinations, occupancy);
	ret -= EvaluateBishops<BLACK>(b.GetPieceTypeBitboard(BB), phase, blackSafeDestinations, occupancy);

	ret += EvaluateRooks<WHITE>(b.GetPieceTypeBitboard(WR), phase, whiteSafeDestinations, occupancy);
	ret -= EvaluateRooks<BLACK>(b.GetPieceTypeBitboard(BR), phase, blackSafeDestinations, occupancy);

	ret += EvaluateQueens<WHITE>(b.GetPieceTypeBitboard(WQ), phase, whiteSafeDestinations, occupancy);
	ret -= EvaluateQueens<BLACK>(b.GetPieceTypeBitboard(BQ), phase, blackSafeDestinations, occupancy);

	ret += EvaluateKings<WHITE>(b.GetPieceTypeBitboard(WK), phase);
	ret -= EvaluateKings<BLACK>(b.GetPieceTypeBitboard(BK), phase);

	ret += (b.GetSideToMove() == WHITE ? SIDE_TO_MOVE_BONUS : (-SIDE_TO_MOVE_BONUS));

	return EvaluatorIface::EvalFullScale * tanh(1e-3f * ret);
}

Score EvaluateMaterial(const Board &b)
{
	Score ret = 0;

	uint32_t WQCount = PopCount(b.GetPieceTypeBitboard(WQ));
	uint32_t WRCount = PopCount(b.GetPieceTypeBitboard(WR));
	uint32_t WBCount = PopCount(b.GetPieceTypeBitboard(WB));
	uint32_t WNCount = PopCount(b.GetPieceTypeBitboard(WN));
	uint32_t WPCount = PopCount(b.GetPieceTypeBitboard(WP));

	uint32_t BQCount = PopCount(b.GetPieceTypeBitboard(BQ));
	uint32_t BRCount = PopCount(b.GetPieceTypeBitboard(BR));
	uint32_t BBCount = PopCount(b.GetPieceTypeBitboard(BB));
	uint32_t BNCount = PopCount(b.GetPieceTypeBitboard(BN));
	uint32_t BPCount = PopCount(b.GetPieceTypeBitboard(BP));

	Phase phase =
		WQCount * Q_PHASE_CONTRIBUTION +
		BQCount * Q_PHASE_CONTRIBUTION +
		WRCount * R_PHASE_CONTRIBUTION +
		BRCount * R_PHASE_CONTRIBUTION +
		WBCount * B_PHASE_CONTRIBUTION +
		BBCount * B_PHASE_CONTRIBUTION +
		WNCount * N_PHASE_CONTRIBUTION +
		BNCount * N_PHASE_CONTRIBUTION +
		WPCount * P_PHASE_CONTRIBUTION +
		BPCount * P_PHASE_CONTRIBUTION;

	if (phase > MAX_PHASE)
	{
		// this can happen on custom positions and in case of promotions
		phase = MAX_PHASE;
	}

	ret += WQCount * ScalePhase(MAT[0][WQ], MAT[1][WQ], phase);
	ret += WRCount * ScalePhase(MAT[0][WR], MAT[1][WR], phase);
	ret += WBCount * ScalePhase(MAT[0][WB], MAT[1][WB], phase);
	ret += WNCount * ScalePhase(MAT[0][WN], MAT[1][WN], phase);
	ret += WPCount * ScalePhase(MAT[0][WP], MAT[1][WP], phase);

	ret -= BQCount * ScalePhase(MAT[0][WQ], MAT[1][WQ], phase);
	ret -= BRCount * ScalePhase(MAT[0][WR], MAT[1][WR], phase);
	ret -= BBCount * ScalePhase(MAT[0][WB], MAT[1][WB], phase);
	ret -= BNCount * ScalePhase(MAT[0][WN], MAT[1][WN], phase);
	ret -= BPCount * ScalePhase(MAT[0][WP], MAT[1][WP], phase);

	return EvaluatorIface::EvalFullScale * tanh(1e-3f * ret);
}

StaticEvaluator gStaticEvaluator;

}
