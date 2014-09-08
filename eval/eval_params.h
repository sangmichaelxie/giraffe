#ifndef EVAL_PARAMS_H
#define EVAL_PARAMS_H

#include <cstdint>

#include "types.h"

namespace Eval
{

static const Score Q_MAT = 1200;
static const Score R_MAT = 600;
static const Score B_MAT = 400;
static const Score N_MAT = 400;
static const Score P_MAT = 100;

static const Phase Q_PHASE_CONTRIBUTION = 4;
static const Phase R_PHASE_CONTRIBUTION = 2;
static const Phase B_PHASE_CONTRIBUTION = 1;
static const Phase N_PHASE_CONTRIBUTION = 1;
static const Phase P_PHASE_CONTRIBUTION = 0;

// max phase without custom positions and promotions
static const Phase MAX_PHASE =
	Q_PHASE_CONTRIBUTION * 2 +
	R_PHASE_CONTRIBUTION * 4 +
	B_PHASE_CONTRIBUTION * 4 +
	N_PHASE_CONTRIBUTION * 4 +
	P_PHASE_CONTRIBUTION * 16;

const float PAWN_PCSQ_MULTIPLIERS[2] = { 1.0f, 2.0f };
const Score PAWN_PCSQ[64] =
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

const float KNIGHT_PCSQ_MULTIPLIERS[2] = { 2.0f, 2.0f };
const Score KNIGHT_PCSQ[64] =
{
	-9, -6, -3, -3, -3, -3, -6, -9,
	-6, -3,  3,  3,  3,  3, -3, -6,
	-3,  3,  7,  7,  7,  7,  3, -3,
	-3,  3,  7,  7,  7,  7,  3, -3,
	-3,  3,  7,  7,  7,  7,  3, -3,
	-3,  3,  7,  7,  7,  7,  3, -3,
	-6, -3,  3,  3,  3,  3, -3, -6,
	-9, -6, -3, -3, -3, -3, -6, -9
};

const float KNIGHT_MOBILITY_MULTIPLIERS[2] = { 2.0f, 1.0f };
const Score KNIGHT_MOBILITY[8] = { -6, -2, 0, 4, 6, 8, 10, 12 };

const float BISHOP_PCSQ_MULTIPLIERS[2] = { 2.0f, 2.0f };
const Score BISHOP_PCSQ[64] =
{
	-3, -3, -3, -3, -3, -3, -3, -3,
	-3,  0,  0,  0,  0,  0,  0, -3,
	-3,  0,  3,  3,  3,  3,  0, -3,
	-3,  0,  3,  5,  5,  3,  0, -3,
	-3,  0,  3,  5,  5,  3,  0, -3,
	-3,  0,  3,  3,  3,  3,  0, -3,
	-3,  0,  0,  0,  0,  0,  0, -3,
	-3, -3, -3, -3, -3, -3, -3, -3
};

const float BISHOP_MOBILITY_MULTIPLIERS[2] = { 2.0f, 1.0f };
const Score BISHOP_MOBILITY[13] = { -4, -2, 0, 1, 2, 3, 4,  5,  6, 7, 8, 9, 10 };

const Score ROOK_PCSQ[2][64] =
{
	{
		0,   0,   2,   2,   2,   2,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		75,	 75,  75,  75,  75,  75,  75,  75,
		0,   0,   0,   0,   0,   0,   0,   0
	},
	{
		0,   0,   2,   2,   2,   2,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,
		10,	 10,  10,  10,  10,  10,  10,  10,
		0,   0,   0,   0,   0,   0,   0,   0
	}
};

const float ROOK_MOBILITY_MULTIPLIERS[2] = { 1.5f, 1.0f };
const Score ROOK_MOBILITY[14] = { -2, -1, 0, 1, 1, 2, 2,  3,  3, 4, 4, 5,  5,  6 };

const float QUEEN_PCSQ_MULTIPLIERS[2] = { 5.0f, 5.0f };
const Score QUEEN_PCSQ[64] =
{
	0,  0,  0,  0,  0,  0,  0,  0,
	0,  1,  1,  1,  1,  1,  1,  0,
	0,  1,  2,  2,  2,  2,  1,  0,
	0,  1,  2,  3,  3,  2,  1,  0,
	0,  1,  2,  3,  3,  2,  1,  0,
	0,  1,  2,  2,  2,  2,  1,  0,
	0,  1,  1,  1,  1,  1,  1,  0,
	0,  0,  0,  0,  0,  0,  0,  0
};

const Score KING_PCSQ[2][64] =
{
	{
		   0,  20,  40, -20, -20, -20,  40,  20,
		-20, -20, -20, -20, -20, -20, -20, -20,
		-40, -40, -40, -40, -40, -40, -40, -40,
		-40, -40, -40, -40, -40, -40, -40, -40,
		-40, -40, -40, -40, -40, -40, -40, -40,
		-40, -40, -40, -40, -40, -40, -40, -40,
		-40, -40, -40, -40, -40, -40, -40, -40,
		-40, -40, -40, -40, -40, -40, -40, -40
	},
	{
		0,  10,  20,  30,  30,  20,  10,   0,
		10,  20,  30,  40,  40,  30,  20,  10,
		20,  30,  40,  50,  50,  40,  30,  20,
		30,  40,  50,  60,  60,  50,  40,  30,
		30,  40,  50,  60,  60,  50,  40,  30,
		20,  30,  40,  50,  50,  40,  30,  20,
		10,  20,  30,  40,  40,  30,  20,  10,
		0,  10,  20,  30,  30,  20,  10,   0
	}
};

}

#endif // EVAL_PARAMS_H
