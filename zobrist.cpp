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
