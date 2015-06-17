#include "ttable.h"

#include <iostream>

TTable::TTable(size_t size)
	: m_size(size), m_currentGeneration(0)
{
	m_data = static_cast<TTEntry *>(malloc(m_size * sizeof(TTEntry)));
}

void TTable::Resize(size_t newSize)
{
	m_size = newSize;

	if (m_data)
	{
		free(m_data);
	}

	m_data = static_cast<TTEntry *>(malloc(m_size * sizeof(TTEntry)));
}

void TTable::Store(uint64_t hash, MoveNoScore bestMove, Score score, int16_t depth, TTEntryType entryType)
{
	TTEntry *slot = &m_data[hash % m_size];

	bool replace = false;

	if (slot->hash != hash)
	{
		// always replace if it's a different position (or the previous position is invalid)
		replace = true;
	}
	else if (entryType == EXACT && slot->entryType != EXACT)
	{
		// always replace with a PV entry, because we use the table to extract PV
		replace = true;
	}
	else if (depth <= 0)
	{
		// don't bother with quiescent nodes
		replace = false;
	}
	else
	{
		// at this point, we have the same kind of nodes (upper/lower bound)
		// we compare depth, but give a penalty for aging so we will eventually replace very old nodes
		int32_t age = m_currentGeneration - slot->birthday;
		if (depth >= (slot->depth - age))
		{
			replace = true;
		}
		else
		{
			replace = false;
		}
	}

	if (replace)
	{
		slot->hash = hash;
		slot->bestMove = bestMove;
		slot->score = score;
		slot->depth = depth;
		slot->entryType = entryType;
		slot->birthday = m_currentGeneration;
	}
}

void TTable::ClearTable()
{
	// we cheat by just incrementing currentGeneration by 1000, so that all entries will be placed on first access
	m_currentGeneration += 1000;
}
