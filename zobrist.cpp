#include "zobrist.h"

#include <random>

void InitializeZobrist()
{
	std::mt19937_64 gen; // using the default seed

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
