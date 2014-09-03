#ifndef SEE_H
#define SEE_H

#include "types.h"
#include "board.h"
#include "eval/eval.h"
#include "move.h"

// returns how good this capture is for the moving side
Score StaticExchangeEvaluation(Board &board, Move mv);

Score StaticExchangeEvaluationImpl(Board &board, Square sq);

bool RunSeeTest(std::string fen, std::string move, Score expectedScore);

void DebugRunSeeTests();

#endif // SEE_H
