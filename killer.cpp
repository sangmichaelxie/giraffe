#include "killer.h"

#include <limits>
#include <algorithm>

Killer::Killer()
{
}

void Killer::Notify(int32_t ply, MoveNoScore move)
{
	if (m_killerMoves.size() < (ply + 1))
	{
		m_killerMoves.resize(ply + 1);

		for (int32_t i = 0; i < NUM_KILLER_MOVES; ++i)
		{
			m_killerMoves[ply].moves[i].first = 0;
			m_killerMoves[ply].moves[i].second = 0;
		}
	}

	// if the move is already in the table, just increment count
	for (int i = 0; i < NUM_KILLER_MOVES; ++i)
	{
		if (m_killerMoves[ply].moves[i].first == move)
		{
			++m_killerMoves[ply].moves[i].second;
			return;
		}
	}

	// otherwise we replace the lowest slot
	int32_t lowestSlot = 0;
	int32_t lowestCount = std::numeric_limits<int32_t>::max();

	for (int32_t i = 0; i < NUM_KILLER_MOVES; ++i)
	{
		if (m_killerMoves[ply].moves[i].second < lowestCount)
		{
			lowestCount = m_killerMoves[ply].moves[i].second;
			lowestSlot = i;
		}
	}

	m_killerMoves[ply].moves[lowestSlot].first = move;
	m_killerMoves[ply].moves[lowestSlot].second = 1;

	std::sort(m_killerMoves[ply].moves, m_killerMoves[ply].moves + NUM_KILLER_MOVES,
			  [](const std::pair<MoveNoScore, int32_t> &a, const std::pair<MoveNoScore, int32_t> &b)
					{
						return a.second > b.second;
					}
			  );
}

Move Killer::GetKiller(int32_t ply, int32_t n)
{
	if (m_killerMoves.size() < (ply + 1))
	{
		return 0;
	}
	else
	{
		return m_killerMoves[ply].moves[n].first;
	}
}

int32_t Killer::GetKillerNum(int32_t ply, Move mv)
{
	if (m_killerMoves.size() < (ply + 1))
	{
		return -1;
	}

	// returns (10 - slot #) if mv is a killer, otherwise -1
	for (int32_t i = 0; i < NUM_KILLER_MOVES; ++i)
	{
		if (m_killerMoves[ply].moves[i].first == mv)
		{
			return (10 - i);
		}
	}

	// returns (10 - slot # - NUM_KILLER_MOVES) if mv is a killer from -2 plies
	if (ply >= 2)
	{
		for (int32_t i = 0; i < NUM_KILLER_MOVES; ++i)
		{
			if (m_killerMoves[ply - 2].moves[i].first == mv)
			{
				return (10 - i - NUM_KILLER_MOVES);
			}
		}
	}

	// returns (10 - slot # - 2xNUM_KILLER_MOVES) if mv is a killer from +2 plies
	if ((ply + 2) < m_killerMoves.size())
	{
		for (int32_t i = 0; i < NUM_KILLER_MOVES; ++i)
		{
			if (m_killerMoves[ply + 2].moves[i].first == mv)
			{
				return (10 - i - 2*NUM_KILLER_MOVES);
			}
		}
	}

	// returns (10 - slot # - 3xNUM_KILLER_MOVES) if mv is a killer from -4 plies
	if (ply >= 4)
	{
		for (int32_t i = 0; i < NUM_KILLER_MOVES; ++i)
		{
			if (m_killerMoves[ply - 4].moves[i].first == mv)
			{
				return (10 - i - 3*NUM_KILLER_MOVES);
			}
		}
	}

	// returns (10 - slot # - 4xNUM_KILLER_MOVES) if mv is a killer from +4 plies
	if ((ply + 4) < m_killerMoves.size())
	{
		for (int32_t i = 0; i < NUM_KILLER_MOVES; ++i)
		{
			if (m_killerMoves[ply + 4].moves[i].first == mv)
			{
				return (10 - i - 4*NUM_KILLER_MOVES);
			}
		}
	}

	return -1;
}

void Killer::MoveMade()
{
	// if a move is made, we have to do 2 things here -
	// 1. decrement all plies
	// 2. half all counts (so old killers will be replaced)

	for (int32_t ply = 1; ply < m_killerMoves.size(); ++ply)
	{
		m_killerMoves[ply - 1] = m_killerMoves[ply];

		for (int32_t i = 0; i < NUM_KILLER_MOVES; ++i)
		{
			m_killerMoves[ply - 1].moves[i].second /= 2;
		}
	}
}
