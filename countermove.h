#ifndef COUNTER_MOVE_H
#define COUNTER_MOVE_H

#include <vector>
#include <utility>

#include "move.h"
#include "board.h"

const static size_t NUM_COUNTER_MOVES = 1;

class CounterMove
{
public:
	CounterMove();

	void Notify(Board &b, Move counterMove);

	// the returned move is not guaranteed to be legal
	Move GetCounterMove(Board &b);

private:
	// color (white = 0), from, to
	Move m_data[2][64][64];
};

#endif // COUNTER_MOVE_H
