/*
	Copyright (C) 2015 Matthew Lai

	Giraffe is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	Giraffe is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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

void TTable::Store(uint64_t hash, Move bestMove, Score score, NodeBudget nodeBudget, TTEntryType entryType)
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
