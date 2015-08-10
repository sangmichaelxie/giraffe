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
	MoveNoScore bestMove;

	// this is set to the table's m_currentGeneration when set
	// and is used to determine how old an entry is
	// m_currentGeneration is incremented after every move on the board;
	int32_t birthday;

	Score score;
	int64_t nodeBudget;
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

	void Store(uint64_t hash, MoveNoScore bestMove, Score score, int64_t nodeBudget, TTEntryType entryType);

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
