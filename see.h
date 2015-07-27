#ifndef SEE_H
#define SEE_H

#include "types.h"
#include "board.h"
#include "eval/eval.h"
#include "move.h"

namespace SEE
{

static const Score SEE_MAT[14] = {
	1500, // WK
	975, // WQ
	500, // WR
	325, // WN
	350, // WB
	100, // WP

	0,
	0,

	1500, // BK
	975, // BQ
	500, // BR
	325, // BN
	350, // BB
	100 // BP
};

// returns how good this capture is for the moving side
Score StaticExchangeEvaluation(Board &board, Move mv);

// returns the value of the largest piece the opponent can place on the square
Score SSEMap(Board &board, Square sq);

Score StaticExchangeEvaluationSq(Board &board, Square sq, bool forced = false);

bool RunSeeTest(std::string fen, std::string move, Score expectedScore);

void DebugRunSeeTests();

}

#endif // SEE_H
