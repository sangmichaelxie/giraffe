#include "eval.h"
#include "types.h"
#include "bit_ops.h"
#include "magic_moves.h"

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
Score EvaluateKnights(uint64_t bb, Phase phase, uint64_t attackedByEnemyPawns)
{
	Score ret = 0;

	while (bb)
	{
		uint32_t idx = Extract(bb);

		uint32_t mobility = PopCount(KNIGHT_ATK[idx] & ~attackedByEnemyPawns);

		ret += ScalePhase(KNIGHT_MOBILITY[mobility] * KNIGHT_MOBILITY_MULTIPLIERS[0],
						  KNIGHT_MOBILITY[mobility] * KNIGHT_MOBILITY_MULTIPLIERS[1], phase);

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(KNIGHT_PCSQ[idx] * KNIGHT_PCSQ_MULTIPLIERS[0], KNIGHT_PCSQ[idx] * KNIGHT_PCSQ_MULTIPLIERS[1], phase);
	}

	return ret;
}

template <Color COLOR>
Score EvaluateBishops(uint64_t bb, Phase phase, uint64_t attackedByEnemyPawns, uint64_t occupancy)
{
	Score ret = 0;

	while (bb)
	{
		uint32_t idx = Extract(bb);

		uint32_t mobility = PopCount(Bmagic(idx, occupancy) & ~attackedByEnemyPawns);

		ret += ScalePhase(BISHOP_MOBILITY[mobility] * BISHOP_MOBILITY_MULTIPLIERS[0],
						  BISHOP_MOBILITY[mobility] * BISHOP_MOBILITY_MULTIPLIERS[1], phase);

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(BISHOP_PCSQ[idx] * BISHOP_PCSQ_MULTIPLIERS[0], BISHOP_PCSQ[idx] * BISHOP_PCSQ_MULTIPLIERS[1], phase);
	}

	return ret;
}

template <Color COLOR>
Score EvaluateRooks(uint64_t bb, Phase phase, uint64_t attackedByEnemyPawns, uint64_t occupancy)
{
	Score ret = 0;

	while (bb)
	{
		uint32_t idx = Extract(bb);

		uint32_t mobility = PopCount(Rmagic(idx, occupancy) & ~attackedByEnemyPawns);

		ret += ScalePhase(ROOK_MOBILITY[mobility] * ROOK_MOBILITY_MULTIPLIERS[0],
						  ROOK_MOBILITY[mobility] * ROOK_MOBILITY_MULTIPLIERS[1], phase);

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(ROOK_PCSQ[0][idx], ROOK_PCSQ[1][idx], phase);
	}

	return ret;
}

template <Color COLOR>
Score EvaluateQueens(uint64_t bb, Phase phase)
{
	Score ret = 0;

	while (bb)
	{
		uint32_t idx = Extract(bb);

		if (COLOR == BLACK)
		{
			idx = FLIP[idx];
		}

		ret += ScalePhase(QUEEN_PCSQ[idx] * QUEEN_PCSQ_MULTIPLIERS[0], QUEEN_PCSQ[idx] * QUEEN_PCSQ_MULTIPLIERS[1], phase);
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

Score Evaluate(const Board &b, Score lowerBound, Score upperBound)
{
	Score ret = 0;

	ret += EvaluateMaterial(b);

	return b.GetSideToMove() == WHITE ? ret : (-ret);
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

	ret += WQCount * Q_MAT;
	ret += WRCount * R_MAT;
	ret += WBCount * B_MAT;
	ret += WNCount * N_MAT;
	ret += WPCount * P_MAT;

	ret -= BQCount * Q_MAT;
	ret -= BRCount * R_MAT;
	ret -= BBCount * B_MAT;
	ret -= BNCount * N_MAT;
	ret -= BPCount * P_MAT;

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

	uint64_t occupancy = b.GetOccupiedBitboard<WHITE>() | b.GetOccupiedBitboard<BLACK>();

	uint64_t attackedByWhitePawns = 0ULL;
	uint64_t attackedByBlackPawns = 0ULL;

	ret += EvaluatePawns<WHITE>(b.GetPieceTypeBitboard(WP), phase, attackedByWhitePawns);
	ret -= EvaluatePawns<BLACK>(b.GetPieceTypeBitboard(BP), phase, attackedByBlackPawns);

	ret += EvaluateKnights<WHITE>(b.GetPieceTypeBitboard(WN), phase, attackedByBlackPawns);
	ret -= EvaluateKnights<BLACK>(b.GetPieceTypeBitboard(BN), phase, attackedByWhitePawns);

	ret += EvaluateBishops<WHITE>(b.GetPieceTypeBitboard(WB), phase, attackedByBlackPawns, occupancy);
	ret -= EvaluateBishops<BLACK>(b.GetPieceTypeBitboard(BB), phase, attackedByWhitePawns, occupancy);

	ret += EvaluateRooks<WHITE>(b.GetPieceTypeBitboard(WR), phase, attackedByBlackPawns, occupancy);
	ret -= EvaluateRooks<BLACK>(b.GetPieceTypeBitboard(BR), phase, attackedByWhitePawns, occupancy);

	ret += EvaluateQueens<WHITE>(b.GetPieceTypeBitboard(WQ), phase);
	ret -= EvaluateQueens<BLACK>(b.GetPieceTypeBitboard(BQ), phase);

	ret += EvaluateKings<WHITE>(b.GetPieceTypeBitboard(WK), phase);
	ret -= EvaluateKings<BLACK>(b.GetPieceTypeBitboard(BK), phase);

	return ret;
}

}
