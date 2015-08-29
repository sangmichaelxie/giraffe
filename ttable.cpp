#include "ttable.h"

#include <iostream>

TTable::TTable(size_t size)
	: m_data(size), m_currentGeneration(0)
{
}

void TTable::Resize(size_t newSize)
{
	m_data.resize(newSize);
}

void TTable::Store(uint64_t hash, MoveNoScore bestMove, Score score, NodeBudget nodeBudget, TTEntryType entryType)
{
	TTEntry *slot = &m_data[hash % m_data.size()];

	bool replace = false;

	if (hash != slot->hash)
	{
		replace = true;
	}
	else
	{
		if (nodeBudget > slot->nodeBudget)
		{
			replace = true;
		}
	}

	if (replace)
	{
		slot->hash = hash;
		slot->bestMove = bestMove;
		slot->score = score;
		slot->nodeBudget = nodeBudget;
		slot->entryType = entryType;
		slot->birthday = m_currentGeneration;
	}
}

void TTable::ClearTable()
{
	// we cheat by just incrementing currentGeneration by 1000, so that all entries will be replaced on first access
	m_currentGeneration += 1000;
}
