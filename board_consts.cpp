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

#include "board_consts.h"
#include "bit_ops.h"

#include <iostream>
#include <iomanip>

uint64_t KING_ATK[64];
uint64_t KNIGHT_ATK[64];

// 0 is white, 1 is black
uint64_t PAWN_ATK[64][2];
uint64_t PAWN_MOVE_1[64][2];
uint64_t PAWN_MOVE_2[64][2];

uint64_t RANK_OF_SQ[64];
uint64_t FILE_OF_SQ[64];
uint64_t ADJACENT_FILES_OF_SQ[64];

uint64_t SqOffset(int32_t sq, int32_t xOffset, int32_t yOffset)
{
	int32_t x = GetX(sq) + xOffset;
	int32_t y = GetY(sq) + yOffset;

	if (Valid(x) && Valid(y))
	{
		return Bit(Sq(x, y));
	}

	return 0ULL;
}

void BoardConstsInit()
{
	// generate all the attack tables
	for (int32_t sq = 0; sq < 64; ++sq)
	{
		KING_ATK[sq] =
			SqOffset(sq, 1, 0) |
			SqOffset(sq, 0, 1) |
			SqOffset(sq, -1, 0) |
			SqOffset(sq, 0, -1) |
			SqOffset(sq, 1, -1) |
			SqOffset(sq, -1, 1) |
			SqOffset(sq, -1, -1) |
			SqOffset(sq, 1, 1);

		KNIGHT_ATK[sq] =
			SqOffset(sq, 2, 1) |
			SqOffset(sq, 2, -1) |
			SqOffset(sq, -2, 1) |
			SqOffset(sq, -2, -1) |
			SqOffset(sq, 1, 2) |
			SqOffset(sq, 1, -2) |
			SqOffset(sq, -1, 2) |
			SqOffset(sq, -1, -2);

		PAWN_ATK[sq][0] = SqOffset(sq, 1, 1) | SqOffset(sq, -1, 1);
		PAWN_ATK[sq][1] = SqOffset(sq, 1, -1) | SqOffset(sq, -1, -1);

		PAWN_MOVE_1[sq][0] = SqOffset(sq, 0, 1);
		PAWN_MOVE_1[sq][1] = SqOffset(sq, 0, -1);

		PAWN_MOVE_2[sq][0] = GetRank(sq) == RANK_2 ? SqOffset(sq, 0, 2) : 0ULL;
		PAWN_MOVE_2[sq][1] = GetRank(sq) == RANK_7 ? SqOffset(sq, 0, -2) : 0ULL;

		RANK_OF_SQ[sq] = RANKS[GetRank(sq)];
		FILE_OF_SQ[sq] = FILES[GetFile(sq)];

		if (GetFile(sq) == 0)
		{
			ADJACENT_FILES_OF_SQ[sq] = FILES[GetFile(sq) + 1];
		}
		else if (GetFile(sq) == 7)
		{
			ADJACENT_FILES_OF_SQ[sq] = FILES[GetFile(sq) - 1];
		}
		else
		{
			ADJACENT_FILES_OF_SQ[sq] = FILES[GetFile(sq) + 1] | FILES[GetFile(sq) - 1];
		}
	}
}

void DebugPrint(uint64_t bb)
{
	for (int32_t y = 7; y >= 0; --y)
	{
		std::cout << static_cast<char>('1' + y) << "| ";
		for (int32_t x = 0; x < 8; ++x)
		{
			std::cout << ((bb & Bit(Sq(x, y))) ? '1' : '0') << ' ';
		}

		std::cout << std::endl;
	}

	std::cout << " -----------------" << std::endl;
	std::cout << "   A B C D E F G H" << std::endl;
}
