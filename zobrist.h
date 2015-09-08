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

#ifndef ZOBRIST_H
#define ZOBRIST_H

#include <cstdint>

#include "types.h"

extern uint64_t PIECES_ZOBRIST[64][PIECE_TYPE_LAST + 1];

extern uint64_t SIDE_TO_MOVE_ZOBRIST;

extern uint64_t EN_PASS_ZOBRIST[64];

extern uint64_t W_SHORT_CASTLE_ZOBRIST;
extern uint64_t W_LONG_CASTLE_ZOBRIST;
extern uint64_t B_SHORT_CASTLE_ZOBRIST;
extern uint64_t B_LONG_CASTLE_ZOBRIST;

void InitializeZobrist();

#endif // ZOBRIST_H
