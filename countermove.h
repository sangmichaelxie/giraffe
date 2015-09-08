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
