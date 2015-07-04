#ifndef MOVEPICKER_H
#define MOVEPICKER_H

#include "board.h"
#include "containers.h"
#include "move.h"
#include "killer.h"

// this class is responsible for generating moves one by one, delaying generation by as long as possible
class MovePicker
{
public:
	enum MovePickerStage
	{
		LIKELY, // hash, queen promotions, winning and equal captures
		NEUTRAL, // killers, other non captures
		UNLIKELY // losing captures, leaving pieces en prise
	};

	// try moves are hash moves and killer moves. they do not have to be
	// valid and will be verified
	MovePicker(Board &b, Move hashMove, Killer &killer, bool isQS, int32_t ply);

	// returns 0 if there are no more moves
	Move GetNextMove(MovePickerStage &stage);

	// returns 0 if there are no more moves
	Move GetNextMove() { MovePickerStage dummy; return GetNextMove(dummy); }

private:
	enum Stage
	{
		HASH_MOVE,
		QUEEN_PROMOTIONS,
		WINNING_EQUAL_CAPTURES,
		KILLERS,
		OTHER_NON_CAPTURES,
		LOSING_CAPTURES // this stage actually includes losing non-captures, and under-promos as well
	};

	void EnterStage_();
	void ExitStage_();

	void AssignSeeScores_(MoveList &ml);
	void RemoveScores_(MoveList &ml);

	Board &m_board;

	Stage m_stage;
	size_t m_i;

	bool m_isQS;

	bool m_firstMoveInStage;

	Move m_hashMove;

	Killer &m_killer;
	KillerMoveList m_killersList;
	int32_t m_ply;

	// move lists are only generated when required
	MoveList m_moveListViolent;
	MoveList m_moveListQuiet;
};

void DebugRunMovePickerTests();

#endif // MOVEPICKER_H
