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

#ifndef MOVE_H
#define MOVE_H

#include <cassert>

#include "types.h"
#include "containers.h"

// what's the maximum number of possible moves from any position?
// 216 seems to be a popular number
static const size_t MAX_LEGAL_MOVES = 256;

typedef uint32_t Move;

// fields
// [3:0] = PieceType
// [9:4] = From
// [15:10] = To
// [19:16] = Castling flags
// [23:20] = promotion piecetype

namespace MoveConstants
{
	const static uint32_t PIECE_TYPE_SHIFT = 0;
	const static uint32_t PIECE_TYPE_MASK = 0xF;
	const static uint32_t FROM_SHIFT = 4;
	const static uint32_t FROM_MASK = 0x3F;
	const static uint32_t TO_SHIFT = 10;
	const static uint32_t TO_MASK = 0x3F;
	const static uint32_t PROMO_SHIFT = 20;
	const static uint32_t PROMO_MASK = 0xF;

	// castling flags are NOT shifted!
	const static uint32_t CASTLE_WHITE_LONG = 1 << 19;
	const static uint32_t CASTLE_WHITE_SHORT = 1 << 18;
	const static uint32_t CASTLE_BLACK_LONG = 1 << 17;
	const static uint32_t CASTLE_BLACK_SHORT = 1 << 16;
	const static uint32_t CASTLE_MASK = CASTLE_WHITE_LONG | CASTLE_WHITE_SHORT | CASTLE_BLACK_LONG | CASTLE_BLACK_SHORT;
}

inline PieceType GetPieceType(Move mv)
{
	return (mv >> MoveConstants::PIECE_TYPE_SHIFT) & MoveConstants::PIECE_TYPE_MASK;
}

inline void SetPieceType(Move &mv, PieceType pt)
{
#ifdef DEBUG
	assert((pt & ~MoveConstants::PIECE_TYPE_MASK) == 0);
	assert(GetPieceType(mv) == 0);
#endif
	mv |= pt << MoveConstants::PIECE_TYPE_SHIFT;
}

inline Square GetFromSquare(Move mv)
{
	return (mv >> MoveConstants::FROM_SHIFT) & MoveConstants::FROM_MASK;
}

inline void SetFromSquare(Move &mv, Square sq)
{
#ifdef DEBUG
	assert((sq & ~MoveConstants::FROM_MASK) == 0);
	assert(GetFromSquare(mv) == 0);
#endif
	mv |= sq << MoveConstants::FROM_SHIFT;
}

inline Square GetToSquare(Move mv)
{
	return (mv >> MoveConstants::TO_SHIFT) & MoveConstants::TO_MASK;
}

inline void SetToSquare(Move &mv, Square sq)
{
#ifdef DEBUG
	assert((sq & ~MoveConstants::TO_MASK) == 0);
	assert(GetToSquare(mv) == 0);
#endif
	mv |= sq << MoveConstants::TO_SHIFT;
}

// 0 means no promotion (0 is the piece type for white king)
inline PieceType GetPromoType(Move mv)
{
	return (mv >> MoveConstants::PROMO_SHIFT) & MoveConstants::PROMO_MASK;
}

inline bool IsPromotion(Move mv)
{
	return GetPromoType(mv) != 0;
}

inline void SetPromoType(Move &mv, PieceType pt)
{
#ifdef DEBUG
	assert((pt & ~MoveConstants::PROMO_MASK) == 0);
	assert(GetPromoType(mv) == 0);
#endif
	mv |= pt << MoveConstants::PROMO_SHIFT;
}

inline bool IsCastling(Move mv)
{
	return mv & MoveConstants::CASTLE_MASK;
}

inline uint32_t GetCastlingType(Move mv)
{
#ifdef DEBUG
	assert(IsCastling(mv));
#endif
	return mv & MoveConstants::CASTLE_MASK;
}

// type should be one of the castling masks
inline void SetCastlingType(Move &mv, uint32_t type)
{
#ifdef DEBUG
	assert((type & ~MoveConstants::CASTLE_MASK) == 0);
	assert((type & MoveConstants::CASTLE_MASK) != 0);
	assert(!IsCastling(mv));
#endif
	mv |= type;
}

typedef FixedVector<Move, MAX_LEGAL_MOVES> MoveList;

#endif // MOVE_H
