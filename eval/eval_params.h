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

#ifndef EVAL_PARAMS_H
#define EVAL_PARAMS_H

#include <cstdint>

#include "types.h"

namespace Eval
{

// most of the values in this file are taken from Stockfish
static const Score MAT[2][14] = {
	{
		10000, // WK (we have a king score here for SEE only)
		2521, // WQ
		1270, // WR
		817, // WN
		836, // WB
		198, // WP

		0,
		0,

		10000, // BK
		2521, // BQ
		1270, // BR
		817, // BN
		836, // BB
		198 // BP
	},
	{
		10000, // WK (we have a king score here for SEE only)
		2558, // WQ
		1278, // WR
		846, // WN
		857, // WB
		258, // WP

		0,
		0,

		10000, // BK
		2558, // BQ
		1278, // BR
		846, // BN
		857, // BB
		258 // BP
	}
};

static const Score MAX_POSITIONAL_SCORE = 150; // approximately how much the positional score can change from 1 move

static const Phase Q_PHASE_CONTRIBUTION = 4;
static const Phase R_PHASE_CONTRIBUTION = 2;
static const Phase B_PHASE_CONTRIBUTION = 1;
static const Phase N_PHASE_CONTRIBUTION = 1;
static const Phase P_PHASE_CONTRIBUTION = 0;

static const float MOBILITY_MULTIPLIERS[2] = { 0.0f, 0.0f };

// max phase without custom positions and promotions
static const Phase MAX_PHASE =
	Q_PHASE_CONTRIBUTION * 2 +
	R_PHASE_CONTRIBUTION * 4 +
	B_PHASE_CONTRIBUTION * 4 +
	N_PHASE_CONTRIBUTION * 4 +
	P_PHASE_CONTRIBUTION * 16;

static const float PAWN_PCSQ_MULTIPLIERS[2] = { 1.0f, 2.0f };
static const Score PAWN_PCSQ[64] =
{
	0,   0,   0,   0,   0,   0,   0,   0,
	0,   0,   0,  -5,  -5,   0,   0,   0,
	1,   3,   2,   4,   4,   2,   3,   1,
	2,   6,   4,   8,   8,   4,   6,   2,
	3,   9,   6,  12,  12,   6,   9,   3,
	4,  12,   8,  16,  16,   8,  12,   4,
	5,  15,  10,  20,  20,  10,  15,   5,
	0,   0,   0,   0,   0,   0,   0,   0
};

static const Score KNIGHT_PCSQ[2][64] =
{
	{
		-144, -109, -85, -73, -73, -85, -109, -144,
		-88, -43, -19, -7, -7, -19, -43, -88,
		-69, -24, 0, 12, 12, 0, -24, -69,
		-28, 17, 41, 53, 53, 41, 17, -28,
		-30, 15, 39, 51, 51, 39, 15, -30,
		-10, 35, 59, 71, 71, 59, 35, -10,
		-64, -19, 5, 17, 17, 5, -19, -64,
		-200, -65, -41, -29, -29, -41, -65, -200,
	},
	{
		-98, -83, -51, -16, -16, -51, -83, -98,
		-68, -53, -21, 14, 14, -21, -53, -68,
		-53, -38, -6, 29, 29, -6, -38, -53,
		-42, -27, 5, 40, 40, 5, -27, -42,
		-42, -27, 5, 40, 40, 5, -27, -42,
		-53, -38, -6, 29, 29, -6, -38, -53,
		-68, -53, -21, 14, 14, -21, -53, -68,
		-98, -83, -51, -16, -16, -51, -83, -98,
	}
};

static const Score KNIGHT_MOBILITY[2][9] = {{ -65, -42, -9, 3, 15, 27, 37, 42, 44 }, { -50, -30, -10, 0, 10, 20, 28, 31, 33 }};

static const Score BISHOP_PCSQ[2][64] =
{
	{
		-54, -27, -34, -43, -43, -34, -27, -54,
		-29, 8, 1, -8, -8, 1, 8, -29,
		-20, 17, 10, 1, 1, 10, 17, -20,
		-19, 18, 11, 2, 2, 11, 18, -19,
		-22, 15, 8, -1, -1, 8, 15, -22,
		-28, 9, 2, -7, -7, 2, 9, -28,
		-32, 5, -2, -11, -11, -2, 5, -32,
		-49, -22, -29, -38, -38, -29, -22, -49,
	},
	{
		-65, -42, -44, -26, -26, -44, -42, -65,
		-43, -20, -22, -4, -4, -22, -20, -43,
		-33, -10, -12, 6, 6, -12, -10, -33,
		-35, -12, -14, 4, 4, -14, -12, -35,
		-35, -12, -14, 4, 4, -14, -12, -35,
		-33, -10, -12, 6, 6, -12, -10, -33,
		-43, -20, -22, -4, -4, -22, -20, -43,
		-65, -42, -44, -26, -26, -44, -42, -65,
	}
};

static const Score BISHOP_MOBILITY[2][14] = { { -52, -28, 6, 20, 34, 48, 60, 68,  74, 77, 80, 82, 84, 86 }, { -47, -23, 1, 15, 29, 43, 55, 63, 68, 72, 75, 77, 84, 86 } };

static const Score BISHOP_PAIR_BONUS[2] = { 50, 75 };

static const Score ROOK_PCSQ[2][64] =
{
	{
		-22, -17, -12, -8, -8, -12, -17, -22,
		-22, -7, -2, 2, 2, -2, -7, -22,
		-22, -7, -2, 2, 2, -2, -7, -22,
		-22, -7, -2, 2, 2, -2, -7, -22,
		-22, -7, -2, 2, 2, -2, -7, -22,
		-22, -7, -2, 2, 2, -2, -7, -22,
		-11, 4, 9, 13, 13, 9, 4, -11,
		-22, -17, -12, -8, -8, -12, -17, -22,
	},
	{
		3, 3, 3, 3, 3, 3, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3,
	}
};

static const Score ROOK_MOBILITY[2][15] = {
	{ -47, -31, -5, 1, 7, 13, 18, 22, 26, 29, 31, 33, 35, 36, 37 },
	{ -53, -26, 0, 16, 32, 48, 64, 80, 96, 109, 115, 119, 122, 123, 124 }};

static const float QUEEN_PCSQ_MULTIPLIERS[2] = { 5.0f, 5.0f };
static const Score QUEEN_PCSQ[2][64] =
{
	{
		-2, -2, -2, -2, -2, -2, -2, -2,
		-2, 8, 8, 8, 8, 8, 8, -2,
		-2, 8, 8, 8, 8, 8, 8, -2,
		-2, 8, 8, 8, 8, 8, 8, -2,
		-2, 8, 8, 8, 8, 8, 8, -2,
		-2, 8, 8, 8, 8, 8, 8, -2,
		-2, 8, 8, 8, 8, 8, 8, -2,
		-2, -2, -2, -2, -2, -2, -2, -2,
	},
	{
		-80, -54, -42, -30, -30, -42, -54, -80,
		-54, -30, -18, -6, -6, -18, -30, -54,
		-42, -18, -6, 6, 6, -6, -18, -42,
		-30, -6, 6, 18, 18, 6, -6, -30,
		-30, -6, 6, 18, 18, 6, -6, -30,
		-42, -18, -6, 6, 6, -6, -18, -42,
		-54, -30, -18, -6, -6, -18, -30, -54,
		-80, -54, -42, -30, -30, -42, -54, -80,
	}
};

static const Score QUEEN_MOBILITY[2][28] = {
	{ -42, -28, -5, 0, 6, 11, 13, 18, 20, 21, 22, 22, 22, 23, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25 },
	{ -40, -23, -7, 0, 10, 19, 29, 38, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41 }};

static const Score KING_PCSQ[2][64] =
{
	{
		298, 332, 273, 225, 225, 273, 332, 298,
		287, 321, 262, 214, 214, 262, 321, 287,
		224, 258, 199, 151, 151, 199, 258, 224,
		196, 230, 171, 123, 123, 171, 230, 196,
		173, 207, 148, 100, 100, 148, 207, 173,
		146, 180, 121, 73, 73, 121, 180, 146,
		119, 153, 94, 46, 46, 94, 153, 119,
		98, 132, 73, 25, 25, 73, 132, 98,
	},
	{
		27, 81, 108, 116, 116, 108, 81, 27,
		74, 128, 155, 163, 163, 155, 128, 74,
		111, 165, 192, 200, 200, 192, 165, 111,
		135, 189, 216, 224, 224, 216, 189, 135,
		135, 189, 216, 224, 224, 216, 189, 135,
		111, 165, 192, 200, 200, 192, 165, 111,
		74, 128, 155, 163, 163, 155, 128, 74,
		27, 81, 108, 116, 116, 108, 81, 27,
	}
};

// give the side to move a small bonus to minimize odd and even effect
static const Score SIDE_TO_MOVE_BONUS = 14;

}

#endif // EVAL_PARAMS_H
