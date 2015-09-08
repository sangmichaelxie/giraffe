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

#ifndef TTABLE_H
#define TTABLE_H

#include <memory>
#include <vector>

#include <cstdint>
#include <cstdlib>

#include "types.h"
#include "move.h"

enum TTEntryType
{
	EXACT,
	LOWERBOUND,
	UPPERBOUND
} __attribute__ ((__packed__));

struct TTEntry
{
	uint64_t hash;
	Move bestMove;

	// this is set to the table's m_currentGeneration when set
	// and is used to determine how old an entry is
	// m_currentGeneration is incremented after every move on the board;
	int32_t birthday;

	Score score;
	NodeBudget nodeBudget;
	TTEntryType entryType;
};

class TTable
{
public:
	TTable(size_t size);

	TTable(const TTable&) = delete;
	TTable &operator=(const TTable&) = delete;

	void Resize(size_t newSize);

	TTEntry *Probe(uint64_t hash)
	{
		size_t idx = hash % m_data.size();
		TTEntry *entry = &m_data[idx];
		if (entry->hash == hash)
		{
			return entry;
		}
		else
		{
			return NULL;
		}
	}

	void Prefetch(uint64_t hash)
	{
		__builtin_prefetch(&m_data[hash % m_data.size()]);
	}

	void Store(uint64_t hash, Move bestMove, Score score, NodeBudget nodeBudget, TTEntryType entryType);

	void AgeTable() { ++m_currentGeneration; }

	// age all entries so any new entry will replace them
	void ClearTable();

	void InvalidateAllEntries()
	{
		for (size_t i = 0; i < m_data.size(); ++i)
		{
			m_data[i].hash = 0;
		}
	}

private:
	std::vector<TTEntry> m_data;

	int32_t m_currentGeneration;
};

#endif // TTABLE_H
