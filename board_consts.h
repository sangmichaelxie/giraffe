#ifndef BOARD_CONSTS_H
#define BOARD_CONSTS_H

#include <cstdint>

#include "types.h"

inline int32_t GetX(Square sq) { return sq % 8; }
inline int32_t GetY(Square sq) { return sq / 8; }
inline int32_t GetRank(Square sq) { return GetY(sq); }
inline int32_t GetFile(Square sq) { return GetX(sq); }
inline Square Sq(int32_t x, int32_t y) { return y * 8 + x; }
inline bool Valid(int32_t x) { return x < 8 && x >= 0; }

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

// we have these as literals to enable constant propagation
const static uint64_t BIT[64] =
{
	0x0000000000000001ULL, 0x0000000000000002ULL, 0x0000000000000004ULL, 0x0000000000000008ULL,
	0x0000000000000010ULL, 0x0000000000000020ULL, 0x0000000000000040ULL, 0x0000000000000080ULL,
	0x0000000000000100ULL, 0x0000000000000200ULL, 0x0000000000000400ULL, 0x0000000000000800ULL,
	0x0000000000001000ULL, 0x0000000000002000ULL, 0x0000000000004000ULL, 0x0000000000008000ULL,
	0x0000000000010000ULL, 0x0000000000020000ULL, 0x0000000000040000ULL, 0x0000000000080000ULL,
	0x0000000000100000ULL, 0x0000000000200000ULL, 0x0000000000400000ULL, 0x0000000000800000ULL,
	0x0000000001000000ULL, 0x0000000002000000ULL, 0x0000000004000000ULL, 0x0000000008000000ULL,
	0x0000000010000000ULL, 0x0000000020000000ULL, 0x0000000040000000ULL, 0x0000000080000000ULL,
	0x0000000100000000ULL, 0x0000000200000000ULL, 0x0000000400000000ULL, 0x0000000800000000ULL,
	0x0000001000000000ULL, 0x0000002000000000ULL, 0x0000004000000000ULL, 0x0000008000000000ULL,
	0x0000010000000000ULL, 0x0000020000000000ULL, 0x0000040000000000ULL, 0x0000080000000000ULL,
	0x0000100000000000ULL, 0x0000200000000000ULL, 0x0000400000000000ULL, 0x0000800000000000ULL,
	0x0001000000000000ULL, 0x0002000000000000ULL, 0x0004000000000000ULL, 0x0008000000000000ULL,
	0x0010000000000000ULL, 0x0020000000000000ULL, 0x0040000000000000ULL, 0x0080000000000000ULL,
	0x0100000000000000ULL, 0x0200000000000000ULL, 0x0400000000000000ULL, 0x0800000000000000ULL,
	0x1000000000000000ULL, 0x2000000000000000ULL, 0x4000000000000000ULL, 0x8000000000000000ULL
};

const static uint64_t ALL = 0xffffffffffffffffULL;

const static uint64_t INVBIT[64] =
{
	0xfffffffffffffffeULL, 0xfffffffffffffffdULL, 0xfffffffffffffffbULL, 0xfffffffffffffff7ULL,
	0xffffffffffffffefULL, 0xffffffffffffffdfULL, 0xffffffffffffffbfULL, 0xffffffffffffff7fULL,
	0xfffffffffffffeffULL, 0xfffffffffffffdffULL, 0xfffffffffffffbffULL, 0xfffffffffffff7ffULL,
	0xffffffffffffefffULL, 0xffffffffffffdfffULL, 0xffffffffffffbfffULL, 0xffffffffffff7fffULL,
	0xfffffffffffeffffULL, 0xfffffffffffdffffULL, 0xfffffffffffbffffULL, 0xfffffffffff7ffffULL,
	0xffffffffffefffffULL, 0xffffffffffdfffffULL, 0xffffffffffbfffffULL, 0xffffffffff7fffffULL,
	0xfffffffffeffffffULL, 0xfffffffffdffffffULL, 0xfffffffffbffffffULL, 0xfffffffff7ffffffULL,
	0xffffffffefffffffULL, 0xffffffffdfffffffULL, 0xffffffffbfffffffULL, 0xffffffff7fffffffULL,
	0xfffffffeffffffffULL, 0xfffffffdffffffffULL, 0xfffffffbffffffffULL, 0xfffffff7ffffffffULL,
	0xffffffefffffffffULL, 0xffffffdfffffffffULL, 0xffffffbfffffffffULL, 0xffffff7fffffffffULL,
	0xfffffeffffffffffULL, 0xfffffdffffffffffULL, 0xfffffbffffffffffULL, 0xfffff7ffffffffffULL,
	0xffffefffffffffffULL, 0xffffdfffffffffffULL, 0xffffbfffffffffffULL, 0xffff7fffffffffffULL,
	0xfffeffffffffffffULL, 0xfffdffffffffffffULL, 0xfffbffffffffffffULL, 0xfff7ffffffffffffULL,
	0xffefffffffffffffULL, 0xffdfffffffffffffULL, 0xffbfffffffffffffULL, 0xff7fffffffffffffULL,
	0xfeffffffffffffffULL, 0xfdffffffffffffffULL, 0xfbffffffffffffffULL, 0xf7ffffffffffffffULL,
	0xefffffffffffffffULL, 0xdfffffffffffffffULL, 0xbfffffffffffffffULL, 0x7fffffffffffffffULL
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
