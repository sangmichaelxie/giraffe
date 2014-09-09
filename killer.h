#ifndef KILLER_H
#define KILLER_H

#include <vector>
#include <utility>

#include "move.h"

static const size_t NUM_KILLER_MOVES = 2;

struct KillerSlot
{
	// these are always sorted by count
	std::pair<MoveNoScore, int32_t> moves[2];
};

class Killer
{
public:
	Killer();

	void Notify(int32_t ply, MoveNoScore move);

	Move GetKiller(int32_t ply, int32_t n);

	// returns (10 - slot #) if mv is a killer, otherwise -1
	int32_t GetKillerNum(int32_t ply, Move mv);

	void MoveMade();

private:
	// this is indexed by ply
	std::vector<KillerSlot> m_killerMoves;
};

#endif // KILLER_H
