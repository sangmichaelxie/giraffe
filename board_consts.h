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

#ifndef BOARD_CONSTS_H
#define BOARD_CONSTS_H

#include <cstdint>

#include "types.h"

// these are initialized on startup
extern uint64_t KING_ATK[64];
extern uint64_t KNIGHT_ATK[64];

// 0 is white, 1 is black
extern uint64_t PAWN_ATK[64][2];
extern uint64_t PAWN_MOVE_1[64][2];

// these bitboards are all 0 except for the starting files, so there is no need to check for that
extern uint64_t PAWN_MOVE_2[64][2];

extern uint64_t RANK_OF_SQ[64];
extern uint64_t FILE_OF_SQ[64];
extern uint64_t ADJACENT_FILES_OF_SQ[64];

const static uint64_t ALL = 0xffffffffffffffffULL;

const static uint64_t BLACK_SQUARES = 0xaa55aa55aa55aa55ULL;

const static uint64_t WHITE_SQUARES = ~BLACK_SQUARES;

const static Square FLIP[64] = {
	 56,  57,  58,  59,  60,  61,  62,  63,
	 48,  49,  50,  51,  52,  53,  54,  55,
	 40,  41,  42,  43,  44,  45,  46,  47,
	 32,  33,  34,  35,  36,  37,  38,  39,
	 24,  25,  26,  27,  28,  29,  30,  31,
	 16,  17,  18,  19,  20,  21,  22,  23,
	  8,   9,  10,  11,  12,  13,  14,  15,
	  0,   1,   2,   3,   4,   5,   6,   7
};

const static uint64_t RANKS[8] = {
	0x00000000000000ffULL,
	0x000000000000ff00ULL,
	0x0000000000ff0000ULL,
	0x00000000ff000000ULL,
	0x000000ff00000000ULL,
	0x0000ff0000000000ULL,
	0x00ff000000000000ULL,
	0xff00000000000000ULL
};

const static uint64_t FILES[8] = {
	0x0101010101010101ULL,
	0x0202020202020202ULL,
	0x0404040404040404ULL,
	0x0808080808080808ULL,
	0x1010101010101010ULL,
	0x2020202020202020ULL,
	0x4040404040404040ULL,
	0x8080808080808080ULL
};

const static Square A1 = 0;
const static Square B1 = 1;
const static Square C1 = 2;
const static Square D1 = 3;
const static Square E1 = 4;
const static Square F1 = 5;
const static Square G1 = 6;
const static Square H1 = 7;
const static Square A2 = 8;
const static Square B2 = 9;
const static Square C2 = 10;
const static Square D2 = 11;
const static Square E2 = 12;
const static Square F2 = 13;
const static Square G2 = 14;
const static Square H2 = 15;
const static Square A3 = 16;
const static Square B3 = 17;
const static Square C3 = 18;
const static Square D3 = 19;
const static Square E3 = 20;
const static Square F3 = 21;
const static Square G3 = 22;
const static Square H3 = 23;
const static Square A4 = 24;
const static Square B4 = 25;
const static Square C4 = 26;
const static Square D4 = 27;
const static Square E4 = 28;
const static Square F4 = 29;
const static Square G4 = 30;
const static Square H4 = 31;
const static Square A5 = 32;
const static Square B5 = 33;
const static Square C5 = 34;
const static Square D5 = 35;
const static Square E5 = 36;
const static Square F5 = 37;
const static Square G5 = 38;
const static Square H5 = 39;
const static Square A6 = 40;
const static Square B6 = 41;
const static Square C6 = 42;
const static Square D6 = 43;
const static Square E6 = 44;
const static Square F6 = 45;
const static Square G6 = 46;
const static Square H6 = 47;
const static Square A7 = 48;
const static Square B7 = 49;
const static Square C7 = 50;
const static Square D7 = 51;
const static Square E7 = 52;
const static Square F7 = 53;
const static Square G7 = 54;
const static Square H7 = 55;
const static Square A8 = 56;
const static Square B8 = 57;
const static Square C8 = 58;
const static Square D8 = 59;
const static Square E8 = 60;
const static Square F8 = 61;
const static Square G8 = 62;
const static Square H8 = 63;

const static int32_t A_FILE = 0;
const static int32_t B_FILE = 1;
const static int32_t C_FILE = 2;
const static int32_t D_FILE = 3;
const static int32_t E_FILE = 4;
const static int32_t F_FILE = 5;
const static int32_t G_FILE = 6;
const static int32_t H_FILE = 7;

const static int32_t RANK_1 = 0;
const static int32_t RANK_2 = 1;
const static int32_t RANK_3 = 2;
const static int32_t RANK_4 = 3;
const static int32_t RANK_5 = 4;
const static int32_t RANK_6 = 5;
const static int32_t RANK_7 = 6;
const static int32_t RANK_8 = 7;

// return the bitboard with only one square set, the square is sq with (xOffset, yOffset)
// if the offseted square is invalid (outside of board), no bit is set
uint64_t SqOffset(int32_t sq, int32_t xOffset, int32_t yOffset);

void BoardConstsInit();

void DebugPrint(uint64_t bb);

#endif // BOARD_CONSTS_H
