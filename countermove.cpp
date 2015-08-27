#include "countermove.h"

#include <limits>
#include <algorithm>

CounterMove::CounterMove()
{
	for (int32_t i = 0; i < 2; ++i)
	{
		for (int32_t j = 0; j < 64; ++j)
		{
			for (int32_t k = 0; k < 64; ++k)
			{
				m_data[i][j][k] = 0;
			}
		}
	}
}

void CounterMove::Notify(Board &b, Move counterMove)
{
	// Board will take care of bound checking etc
	Optional<Move> lastMove = b.GetMoveFromLast(0);

	if (!lastMove)
	{
		return;
	}

	// for null moves, we will have 0 and 0 for from and to, that's OK
	Square from = GetFromSquare(*lastMove);
	Square to = GetToSquare(*lastMove);
	Color stm = b.GetSideToMove();

	m_data[stm == WHITE ? 0 : 1][from][to] = counterMove;
}

Move CounterMove::GetCounterMove(Board &b)
{
	// Board will take care of bound checking etc
	Optional<Move> lastMove = b.GetMoveFromLast(0);

	if (!lastMove)
	{
		return 0;
	}

	// for null moves, we will have 0 and 0 for from and to, that's OK
	Square from = GetFromSquare(*lastMove);
	Square to = GetToSquare(*lastMove);
	Color stm = b.GetSideToMove();

	return m_data[stm == WHITE ? 0 : 1][from][to];
}
