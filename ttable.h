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
	int16_t depth;
	TTEntryType entryType;

};

class TTable
{
public:
	TTable(size_t size);
	~TTable() { free(m_data); }

	void Resize(size_t newSize);

	TTEntry *Probe(uint64_t hash)
	{
		size_t idx = hash % m_size;
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

	void Store(uint64_t hash, MoveNoScore bestMove, Score score, int16_t depth, TTEntryType entryType);

	void AgeTable() { ++m_currentGeneration; }

	// invalidate all entries
	void ClearTable();

private:
	TTEntry *m_data;
	size_t m_size;

	int32_t m_currentGeneration;
};

#endif // TTABLE_H
