#ifndef EVAL_H
#define EVAL_H

#include <limits>

#include <cstdint>

#include "board.h"

typedef int32_t Score;

const static Score SCORE_MAX = std::numeric_limits<Score>::max();
const static Score SCORE_MIN = std::numeric_limits<Score>::min();

namespace Eval
{

// returns score for side to move
Score Evaluate(const Board &b, Score lowerBound, Score upperBound);

// returns score for white
Score EvaluateMaterial(const Board &b);

}

#endif // EVAL_H
