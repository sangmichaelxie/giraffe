#ifndef KILLER_H
#define KILLER_H

#include <vector>
#include <utility>

#include "move.h"

static const size_t NUM_KILLER_MOVES_PER_PLY = 2;

static const size_t NUM_KILLER_MOVES = 6; // 2 from current ply, 2 from ply-2, 2 from ply+2

typedef FixedVector<Move, NUM_KILLER_MOVES> KillerMoveList;

struct KillerSlot
{
	Move moves[NUM_KILLER_MOVES_PER_PLY];
};

class Killer
{
public:
	Killer();

	void Notify(int32_t ply, MoveNoScore move);

	void GetKillers(KillerMoveList &moveList, int32_t ply);

	void MoveMade();

private:
	// this is indexed by ply
	std::vector<KillerSlot> m_killerMoves;
};

#endif // KILLER_H
