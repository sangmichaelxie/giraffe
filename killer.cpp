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
			m_killerMoves[ply].moves[i] = 0;
			m_killerMoves[ply].moves[i] = 0;
		}
	}

	// if the move is already in the table, we don't need to do anything
	if (m_killerMoves[ply].moves[0] == move)
	{
		return;
	}

	// otherwise, push everything down one slot
	for (int64_t i = (NUM_KILLER_MOVES_PER_PLY - 1); i >= 0; --i)
	{
		if (i == 0)
		{
			m_killerMoves[ply].moves[i] = move;
		}
		else
		{
			m_killerMoves[ply].moves[i] = m_killerMoves[ply].moves[i - 1];
		}
	}
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
		moveList.PushBack(m_killerMoves[ply].moves[i]);
	}

	// moves from the ply-2
	if (ply >= 2)
	{
		for (size_t i = 0; i < NUM_KILLER_MOVES_PER_PLY; ++i)
		{
			moveList.PushBack(m_killerMoves[ply - 2].moves[i]);
		}
	}

	// moves from the ply+2
	if ((static_cast<size_t>(ply) + 2) < m_killerMoves.size())
	{
		for (size_t i = 0; i < NUM_KILLER_MOVES_PER_PLY; ++i)
		{
			moveList.PushBack(m_killerMoves[ply + 2].moves[i]);
		}
	}
}

void Killer::MoveMade()
{
	// if a move is made, we decrement all plies

	for (size_t ply = 1; ply < m_killerMoves.size(); ++ply)
	{
		m_killerMoves[ply - 1] = m_killerMoves[ply];
	}
}
