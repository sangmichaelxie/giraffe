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

#ifndef KILLER_H
#define KILLER_H

#include <vector>
#include <utility>

#include "move.h"

static const size_t NUM_KILLER_MOVES_PER_PLY = 2;

static const size_t NUM_KILLER_MOVES = 6; // 2 from current ply, 2 from ply-2, 2 from ply+2

typedef FixedVector<Move, NUM_KILLER_MOVES> KillerMoveList;

struct KillerSlot
{
	Move moves[NUM_KILLER_MOVES_PER_PLY];
};

class Killer
{
public:
	Killer();

	void Notify(int32_t ply, Move move);

	void GetKillers(KillerMoveList &moveList, int32_t ply);

	void MoveMade();

private:
	// this is indexed by ply
	std::vector<KillerSlot> m_killerMoves;
};

#endif // KILLER_H
