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

#ifndef TYPES_H
#define TYPES_H

#include <string>

#include <cstdint>

typedef uint32_t Square;

inline int32_t GetX(Square sq) { return sq % 8; }
inline int32_t GetY(Square sq) { return sq / 8; }
inline int32_t GetRank(Square sq) { return GetY(sq); }
inline int32_t GetFile(Square sq) { return GetX(sq); }
inline Square Sq(int32_t x, int32_t y) { return y * 8 + x; }
inline bool Valid(int32_t x) { return x < 8 && x >= 0; }

inline std::string SquareToString(Square sq)
{
	char rank = '1' + (sq / 8);
	char file = 'a' + (sq % 8);

	if (sq == 0xff)
	{
		return std::string("-");
	}
	else
	{
		return std::string(&file, 1) + std::string(&rank, 1);
	}
}

inline int32_t GetDiag0(Square sq)
{
	static const uint32_t diag[64] =
	{
		0, 1, 2, 3, 4, 5, 6, 7,
		1, 2, 3, 4, 5, 6, 7, 8,
		2, 3, 4, 5, 6, 7, 8, 9,
		3, 4, 5, 6, 7, 8, 9, 10,
		4, 5, 6, 7, 8, 9, 10, 11,
		5, 6, 7, 8, 9, 10, 11, 12,
		6, 7, 8, 9, 10, 11, 12, 13,
		7, 8, 9, 10, 11, 12, 13, 14
	};

	return diag[GetY(sq)*8 + GetX(sq)];
}

inline int32_t GetDiag1(Square sq)
{
	static const uint32_t diag[64] =
	{
		0, 1, 2, 3, 4, 5, 6, 7,
		1, 2, 3, 4, 5, 6, 7, 8,
		2, 3, 4, 5, 6, 7, 8, 9,
		3, 4, 5, 6, 7, 8, 9, 10,
		4, 5, 6, 7, 8, 9, 10, 11,
		5, 6, 7, 8, 9, 10, 11, 12,
		6, 7, 8, 9, 10, 11, 12, 13,
		7, 8, 9, 10, 11, 12, 13, 14
	};

	return diag[GetY(sq)*8 + (7 - GetX(sq))];
}

typedef uint32_t Color;

typedef int16_t Score;
typedef int32_t Phase;

const static Color WHITE = 0x0;
const static Color BLACK = 0x8;
const static uint32_t COLOR_MASK = 0x8;

inline int32_t GetEqY(Square sq, Color c) { int32_t y = GetY(sq); return (c == WHITE) ? y : 7 - y; }

typedef uint32_t PieceType;

const static uint32_t NUM_PIECETYPES = 12;
const static uint32_t PIECE_TYPE_INDICES[] = { 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd };

const static uint32_t COMPRESS_PT_IDX[14] = { 0, 1, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9, 10, 11 };

// remember to update material tables if these assignments change

const static PieceType WK = 0x0; // 0b0000
const static PieceType WQ = 0x1; // 0b0001
const static PieceType WR = 0x2; // 0b0010
const static PieceType WN = 0x3; // 0b0011
const static PieceType WB = 0x4; // 0b0100
const static PieceType WP = 0x5; // 0b0101

// 0x6 is used by the white_occupied bitboard

// the empty value is put here to allow faster move application (no special case for empty squares on board updates)
// the value is never used
const static PieceType EMPTY = 0x7;

const static PieceType BK = 0x8; // 0b1000
const static PieceType BQ = 0x9; // 0b1001
const static PieceType BR = 0xa; // 0b1010
const static PieceType BN = 0xb; // 0b1011
const static PieceType BB = 0xc; // 0b1100
const static PieceType BP = 0xd; // 0b1101

// colour-neutral piece types
const static PieceType K = WK; // 0b0000
const static PieceType Q = WQ; // 0b0001
const static PieceType R = WR; // 0b0010
const static PieceType N = WN; // 0b0011
const static PieceType B = WB; // 0b0100
const static PieceType P = WP; // 0b0101

// 0xe is used by the black_occupied bitboard

const static PieceType PIECE_TYPE_LAST = BP;

inline Color GetColor(PieceType pt)
{
	return pt & COLOR_MASK;
}

inline PieceType StripColor(PieceType pt)
{
	return pt & ~COLOR_MASK;
}

inline char PieceTypeToChar(PieceType pt)
{
	switch (pt)
	{
	case WK:
		return 'K';
	case WQ:
		return 'Q';
	case WB:
		return 'B';
	case WN:
		return 'N';
	case WR:
		return 'R';
	case WP:
		return 'P';

	case BK:
		return 'k';
	case BQ:
		return 'q';
	case BB:
		return 'b';
	case BN:
		return 'n';
	case BR:
		return 'r';
	case BP:
		return 'p';

	case EMPTY:
		return ' ';
	}

	return '?';
}

const static size_t KB = 1024;
const static size_t MB = 1024*KB;

// a score of MATE_MOVING_SIDE means the opponent (of the moving side) is mated on the board
const static Score MATE_MOVING_SIDE = 30000;

// a score of MATE_OPPONENT_SIDE means the moving side is mated on the board
const static Score MATE_OPPONENT_SIDE = -30000;

const static Score MATE_MOVING_SIDE_THRESHOLD = 20000;
const static Score MATE_OPPONENT_SIDE_THRESHOLD = -20000;

// when these mating scores are propagated up, they are adjusted by distance to mate
inline void AdjustIfMateScore(Score &score)
{
	if (score > MATE_MOVING_SIDE_THRESHOLD)
	{
		--score;
	}
	else if (score < MATE_OPPONENT_SIDE_THRESHOLD)
	{
		++score;
	}
}

inline bool IsMateScore(Score score)
{
	return score > MATE_MOVING_SIDE_THRESHOLD || score < MATE_OPPONENT_SIDE_THRESHOLD;
}

inline Score MakeWinningScore(int32_t plies)
{
	return MATE_MOVING_SIDE - plies;
}

inline Score MakeLosingScore(int32_t plies)
{
	return MATE_OPPONENT_SIDE + plies;
}

// this is basically std::experimental::optional, but we don't want to switch to C++1y just for this
template <typename T>
class Optional
{
public:
	Optional() : m_valid(false) {}
	Optional(T x) : m_val(x), m_valid(true) {}
	operator bool() const { return m_valid; }

	// assignments should be done through operator= so m_valid can be updated
	const T &operator*() { return m_val; }

	Optional &operator=(const T &val) { m_val = val; m_valid = true; return *this; }

private:
	T m_val;
	bool m_valid;
};

typedef uint64_t NodeBudget;

#endif // TYPES_H
