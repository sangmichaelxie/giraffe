#include "eval.h"
#include "types.h"
#include "bit_ops.h"

namespace Eval
{

Score EvaluatePawn(uint64_t bb, PieceType pt)
{
	const Score PCSQ[64]
	{
		0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,  -5,  -5,   0,   0,   0,
		1,   3,   2,   4,   4,   2,   3,   1,
		2,   6,   4,   8,   8,   4,   6,   2,
		3,   9,   6,  12,  12,   6,   9,   3,
		4,  12,   8,  16,  16,   8,  12,   4,
		5,  15,  10,  20,  20,  10,  15,   5,
		0,   0,   0,   0,   0,   0,   0,   0
	};

	Score ret = 0;

	if (pt == WP)
	{
		while (bb)
		{
			uint32_t idx = Extract(bb);
			ret += PCSQ[idx];
		}
	}
	else
	{
		while (bb)
		{
			uint32_t idx = Extract(bb);
			ret -= PCSQ[FLIP[idx]];
		}
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

	ret += PopCount(b.GetPieceTypeBitboard(WQ)) * 1200;
	ret += PopCount(b.GetPieceTypeBitboard(WR)) * 600;
	ret += PopCount(b.GetPieceTypeBitboard(WB)) * 400;
	ret += PopCount(b.GetPieceTypeBitboard(WN)) * 400;
	ret += PopCount(b.GetPieceTypeBitboard(WP)) * 100;

	ret -= PopCount(b.GetPieceTypeBitboard(BQ)) * 1200;
	ret -= PopCount(b.GetPieceTypeBitboard(BR)) * 600;
	ret -= PopCount(b.GetPieceTypeBitboard(BB)) * 400;
	ret -= PopCount(b.GetPieceTypeBitboard(BN)) * 400;
	ret -= PopCount(b.GetPieceTypeBitboard(BP)) * 100;

	ret += EvaluatePawn(b.GetPieceTypeBitboard(WP), WP);
	ret += EvaluatePawn(b.GetPieceTypeBitboard(BP), BP);

	return ret;
}

}
