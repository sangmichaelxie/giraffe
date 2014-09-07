#include "see.h"

#include <cstdint>

static const Score SEE_VALUES[] = {
	100000, // WK
	900, // WQ
	500, // WR
	300, // WN
	300, // WB
	100, // WP

	0,
	0,

	100000, // BK
	900, // BQ
	500, // BR
	300, // BN
	300, // BB
	100, // BP
};

// best tactical result for the moving side
Score StaticExchangeEvaluation(Board &board, Move mv)
{
	if (!board.IsSeeEligible(mv))
	{
		// if this position is not a legal capture, return 0 so that it doesn't get searched in Qsearch
		return 0;
	}

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
	Score ret = SEE_VALUES[capturedPT] - StaticExchangeEvaluationImpl(board, to);

	board.UndoMoveSee();

	return ret;
}

Score StaticExchangeEvaluationImpl(Board &board, Square sq)
{
	Score ret = 0;

	PieceType pt;
	Square from;

	bool hasMoreCapture = board.GenerateSmallestCaptureSee(pt, from, sq);

	if (hasMoreCapture)
	{
		PieceType capturedPT = board.ApplyMoveSee(pt, from, sq);
		ret = std::max(0, SEE_VALUES[capturedPT] - StaticExchangeEvaluationImpl(board, sq));
		board.UndoMoveSee();
	}

	return ret;
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

	std::cout << "Passed" << std::endl;

	return true;
}

void DebugRunSeeTests()
{
	// basic white capture, Rxd5
	if (!RunSeeTest("7k/8/8/3p4/8/3R4/8/K7 w - - 0 1", "d3d5", 100)) { abort(); }

	// basic black capture, exf5
	if (!RunSeeTest("7k/8/8/4p3/5R2/8/8/K7 b - - 0 1", "e5f4", 500)) { abort(); }

	// simple exchange, exf4 Rxf4
	if (!RunSeeTest("7k/8/8/4p3/5R1R/8/8/K7 b - - 0 1", "e5f4", 400)) { abort(); }

	// decide to not capture, exf4
	if (!RunSeeTest("7k/8/8/4p3/5R2/8/8/K7 b - - 0 1", "e5f4", 500)) { abort(); }

	// decide to not recapture due to discovered attacker, Rxe6
	if (!RunSeeTest("7k/4q3/4q3/8/4R3/4R3/8/K7 w - - 0 1", "e4e6", 900)) { abort(); }

	// recapture without the discovered attacker, Rxe6 Qxe6
	if (!RunSeeTest("7k/4q3/4q3/8/4R3/8/8/K7 w - - 0 1", "e4e6", 400)) { abort(); }

	// complex capture sequence, cxd4 exd4 Nxd4
	if (!RunSeeTest("4q2k/3q2b1/8/2p5/3P4/4P3/3Rn3/K2R4 b - - 0 1", "c5d4", 100)) { abort(); }

	// similar situation, but less defender
	if (!RunSeeTest("4q2k/3q4/8/2p5/3P4/4P3/3R4/K2R4 b - - 0 1", "c5d4", 0)) { abort(); }

	// queen defender blocked by pawn, cxd4
	if (!RunSeeTest("7k/q7/8/2p5/3P4/8/3R4/6K1 b - - 0 1", "c5d4", 100)) { abort(); }

	// bad capture by black, Nxd4 Rxd4
	if (!RunSeeTest("7k/q7/2n5/8/3P4/8/3R4/3R2K1 b - - 0 1", "c6d4", -200)) { abort(); }

	// bad capture by white, Rxd4 Nxd4
	if (!RunSeeTest("7k/q7/2n5/8/3p4/8/3R4/3R2K1 w - - 0 1", "d2d4", -400)) { abort(); }
}