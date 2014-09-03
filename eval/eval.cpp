#include "eval.h"
#include "types.h"
#include "bit_ops.h"

namespace Eval
{

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

	return ret;
}

}
