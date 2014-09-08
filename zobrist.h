#ifndef ZOBRIST_H
#define ZOBRIST_H

#include <cstdint>

#include "types.h"

uint64_t PIECES_ZOBRIST[64][PIECE_TYPE_LAST + 1];

uint64_t SIDE_TO_MOVE_ZOBRIST;

uint64_t EN_PASS_ZOBRIST[64];

uint64_t W_SHORT_CASTLE_ZOBRIST;
uint64_t W_LONG_CASTLE_ZOBRIST;
uint64_t B_SHORT_CASTLE_ZOBRIST;
uint64_t B_LONG_CASTLE_ZOBRIST;

void InitializeZobrist();

#endif // ZOBRIST_H
