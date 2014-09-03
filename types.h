#ifndef TYPES_H
#define TYPES_H

#include <string>

#include <cstdint>

typedef uint32_t Square;
typedef uint32_t Color;

const static Color WHITE = 0x0;
const static Color BLACK = 0x8;
const static uint32_t COLOR_MASK = 0x8;

typedef uint32_t PieceType;

const static uint32_t NUM_PIECETYPES = 12;
const static uint32_t PIECE_TYPE_INDICES[] = { 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd };

// remember to update SEE_VALUES if these assignments change

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

#endif // TYPES_H
