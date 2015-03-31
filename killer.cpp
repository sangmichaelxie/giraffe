#include "killer.h"

#include <limits>
#include <algorithm>

Killer::Killer()
{
}

void Killer::Notify(int32_t ply, MoveNoScore move)
{
	if (m_killerMoves.size() < (static_cast<size_t>(ply) + 1))
	{
		m_killerMoves.resize(ply + 1);

		for (size_t i = 0; i < NUM_KILLER_MOVES_PER_PLY; ++i)
		{
			m_killerMoves[ply].moves[i].first = 0;
			m_killerMoves[ply].moves[i].second = 0;
		}
	}

	// if the move is already in the table, just increment count
	for (size_t i = 0; i < NUM_KILLER_MOVES_PER_PLY; ++i)
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

	for (size_t i = 0; i < NUM_KILLER_MOVES_PER_PLY; ++i)
	{
		if (m_killerMoves[ply].moves[i].second < lowestCount)
		{
			lowestCount = m_killerMoves[ply].moves[i].second;
			lowestSlot = i;
		}
	}

	m_killerMoves[ply].moves[lowestSlot].first = move;
	m_killerMoves[ply].moves[lowestSlot].second = 1;

	std::sort(m_killerMoves[ply].moves, m_killerMoves[ply].moves + NUM_KILLER_MOVES_PER_PLY,
			  [](const std::pair<MoveNoScore, int32_t> &a, const std::pair<MoveNoScore, int32_t> &b)
					{
						return a.second > b.second;
					}
			  );
}

void Killer::GetKillers(KillerMoveList &moveList, int32_t ply)
{
	moveList.Clear();

	if (m_killerMoves.size() < (static_cast<size_t>(ply) + 1))
	{
		return;
	}

	// moves from the current ply
	for (size_t i = 0; i < NUM_KILLER_MOVES_PER_PLY; ++i)
	{
		moveList.PushBack(m_killerMoves[ply].moves[i].first);
	}

	// moves from the ply-2
	if (ply >= 2)
	{
		for (size_t i = 0; i < NUM_KILLER_MOVES_PER_PLY; ++i)
		{
			moveList.PushBack(m_killerMoves[ply - 2].moves[i].first);
		}
	}

	// moves from the ply+2
	if ((static_cast<size_t>(ply) + 2) < m_killerMoves.size())
	{
		for (size_t i = 0; i < NUM_KILLER_MOVES_PER_PLY; ++i)
		{
			moveList.PushBack(m_killerMoves[ply + 2].moves[i].first);
		}
	}
}

void Killer::MoveMade()
{
	// if a move is made, we have to do 2 things here -
	// 1. decrement all plies
	// 2. half all counts (so old killers will be replaced)

	for (size_t ply = 1; ply < m_killerMoves.size(); ++ply)
	{
		m_killerMoves[ply - 1] = m_killerMoves[ply];

		for (size_t i = 0; i < NUM_KILLER_MOVES_PER_PLY; ++i)
		{
			m_killerMoves[ply - 1].moves[i].second /= 2;
		}
	}
}
