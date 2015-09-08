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

#include "zobrist.h"

#include <random>

uint64_t PIECES_ZOBRIST[64][PIECE_TYPE_LAST + 1];

uint64_t SIDE_TO_MOVE_ZOBRIST;

uint64_t EN_PASS_ZOBRIST[64];

uint64_t W_SHORT_CASTLE_ZOBRIST;
uint64_t W_LONG_CASTLE_ZOBRIST;
uint64_t B_SHORT_CASTLE_ZOBRIST;
uint64_t B_LONG_CASTLE_ZOBRIST;

void InitializeZobrist()
{
	std::mt19937_64 gen(53820873); // using the default seed

	for (int32_t sq = 0; sq < 64; ++sq)
	{
		for (PieceType pt = 0; pt <= PIECE_TYPE_LAST; ++pt)
		{
			PIECES_ZOBRIST[sq][pt] = gen();
		}

		EN_PASS_ZOBRIST[sq] = gen();
	}

	SIDE_TO_MOVE_ZOBRIST = gen();

	W_SHORT_CASTLE_ZOBRIST = gen();
	W_LONG_CASTLE_ZOBRIST = gen();
	B_SHORT_CASTLE_ZOBRIST = gen();
	B_LONG_CASTLE_ZOBRIST = gen();
}
