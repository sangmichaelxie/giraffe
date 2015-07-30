#include "movepicker.h"

#include <algorithm>
#include <iostream>

#include <cassert>
#include <cstdlib>

#include "see.h"

MovePicker::MovePicker(Board &b, Move hashMove, Killer &killer, bool isQS, int32_t ply)
	: m_board(b),
	  m_stage(MovePicker::HASH_MOVE),
	  m_i(0),
	  m_isQS(isQS),
	  m_firstMoveInStage(false),
	  m_hashMove(hashMove),
	  m_killer(killer),
	  m_ply(ply)
{
}

Move MovePicker::GetNextMove(MovePickerStage &stage)
{
	if (m_firstMoveInStage)
	{
		EnterStage_();
	}

	Move ret;

	switch (m_stage)
	{
	case HASH_MOVE:
		if (m_hashMove && m_board.CheckPseudoLegal(m_hashMove) && (!m_isQS || m_board.IsViolent(m_hashMove)))
		{
			// we have to exit stage here because we are not clearing m_hashMove
			// we cannot get back in here again otherwise we'll return the hash move again
			ExitStage_();
			ret = m_hashMove;
			Score seeScore = SEE::StaticExchangeEvaluation(m_board, ret);
			SetScoreBiased(ret, seeScore);
			stage = LIKELY;
			return ret;
		}
		ExitStage_();
		return GetNextMove();
		break;
	case QUEEN_PROMOTIONS:
		for (; m_i < m_moveListViolent.GetSize(); ++m_i)
		{
			if (m_moveListViolent[m_i])
			{
				if (ClearScore(m_moveListViolent[m_i]) == m_hashMove)
				{
					m_moveListViolent[m_i] = 0;
					continue;
				}

				if (GetPromoType(m_moveListViolent[m_i]) != 0) // we will only get queen promotions in violent moves
				{
					// don't promote and get captured right away, with no compensation
					Score seeScore = SEE::StaticExchangeEvaluation(m_board, m_moveListViolent[m_i]);

					if (seeScore >= 0)
					{
						ret = m_moveListViolent[m_i];
						m_moveListViolent[m_i++] = 0;
						SetScoreBiased(ret, seeScore);
						stage = LIKELY;
						return ret;
					}
				}
			}
		}
		ExitStage_();
		return GetNextMove();
		break;
	case WINNING_EQUAL_CAPTURES:
		// the list should be sorted from least valuable attacker to most valuable already
		// so all we have to do here is making sure SEE returns positive
		for (; m_i < m_moveListViolent.GetSize(); ++m_i)
		{
			if (m_moveListViolent[m_i])
			{
				if (ClearScore(m_moveListViolent[m_i]) == m_hashMove)
				{
					m_moveListViolent[m_i] = 0;
					continue;
				}

				Score seeScore = SEE::StaticExchangeEvaluation(m_board, m_moveListViolent[m_i]);

				if (seeScore >= 0)
				{
					ret = m_moveListViolent[m_i];
					m_moveListViolent[m_i++] = 0;
					SetScoreBiased(ret, seeScore);
					stage = LIKELY;
					return ret;
				}
			}
		}

		ExitStage_();

		if (m_isQS)
		{
			// if we are in QS, this is it!
			return 0;
		}
		else
		{
			return GetNextMove();
		}
		break;
	case KILLERS:
		for (; m_i < m_killersList.GetSize(); ++m_i)
		{
			if (m_killersList[m_i])
			{
				// we only have to check hash move here since queen promotions and winning captures
				// are violent moves, and cannot be killers
				if (m_killersList[m_i] == m_hashMove)
				{
					m_killersList[m_i] = 0;
					continue;
				}

				// we also have to verify that the move is non-violent
				// because of the way we encode moves
				// it's possible for a move to be violent in one position, but not another
				if (m_board.IsViolent(m_killersList[m_i]))
				{
					m_killersList[m_i] = 0;
					continue;
				}

				// we can have repeated killers (from different plies), so we take out all other same killers
				for (size_t i = 0; i < m_killersList.GetSize(); ++i)
				{
					if (m_killersList[i] == m_killersList[m_i] && i != m_i)
					{
						m_killersList[i] = 0;
					}
				}

				if (m_board.CheckPseudoLegal(m_killersList[m_i]))
				{
					ret = m_killersList[m_i++];
					Score seeScore = SEE::StaticExchangeEvaluation(m_board, ret);
					SetScoreBiased(ret, seeScore);
					stage = NEUTRAL;
					return ret;
				}
			}
		}
		ExitStage_();
		return GetNextMove();
		break;
	case OTHER_NON_CAPTURES:
		// this stage actually includes under-promotions as well (including capturing under-promotions)
		for (; m_i < m_moveListQuiet.GetSize(); ++m_i)
		{
			if (m_moveListQuiet[m_i])
			{
				if (m_moveListQuiet[m_i] == m_hashMove || m_killersList.Exists(ClearScore(m_moveListQuiet[m_i])))
				{
					continue;
				}

				// don't worry about underpromotions
				// leave them to losing captures stage, since they are very rarely good
				if (GetPromoType(m_moveListQuiet[m_i]) != 0)
				{
					m_moveListViolent.PushBack(m_moveListQuiet[m_i]);
					continue;
				}

				// if the move is losing according to SEE, put it in losing captures list
				// here we have to guarantee that this move is not a killer, because
				// in the next stage we don't check that (since captures can't be killers)
				Score seeScore = SEE::StaticExchangeEvaluation(m_board, m_moveListQuiet[m_i]);

				if (seeScore < 0)
				{
					m_moveListViolent.PushBack(m_moveListQuiet[m_i]);
					continue;
				}

				ret = m_moveListQuiet[m_i++];
				SetScoreBiased(ret, seeScore);
				stage = NEUTRAL;
				return ret;
			}
		}
		ExitStage_();
		return GetNextMove();
		break;
	case LOSING_CAPTURES:
		// this is just all the remaining violent moves
		for (; m_i < m_moveListViolent.GetSize(); ++m_i)
		{
			if (m_moveListViolent[m_i])
			{
				if (m_moveListViolent[m_i] == m_hashMove)
				{
					continue;
				}

				ret = m_moveListViolent[m_i++];
				Score seeScore = SEE::StaticExchangeEvaluation(m_board, ret);
				SetScoreBiased(ret, seeScore);
				stage = UNLIKELY;
				return ret;
			}
		}
		ExitStage_();
		// all done!
		return 0;
		break;
	default:
		assert(!"Unknown move picker stage!");
		return 0;
		break;
	}
}

void MovePicker::EnterStage_()
{
	m_firstMoveInStage = false;

	switch (m_stage)
	{
	case QUEEN_PROMOTIONS:
		// on entering this stage, we have to generate all violent moves, and sort them by SEE
		m_board.GenerateAllLegalMoves<Board::VIOLENT>(m_moveListViolent);

		m_i = 0;
		break;
	case WINNING_EQUAL_CAPTURES:
		m_i = 0;
		break;
	case KILLERS:
		m_killer.GetKillers(m_killersList, m_ply);
		m_i = 0;
		break;
	case OTHER_NON_CAPTURES:
		// at this point we have to generate quiet moves
		m_board.GenerateAllLegalMoves<Board::QUIET>(m_moveListQuiet);

		m_i = 0;
		break;
	case LOSING_CAPTURES:
		m_i = 0;
		break;
	default:
		break;
	}
}

void MovePicker::ExitStage_()
{
	m_firstMoveInStage = true;

	switch (m_stage)
	{
	case HASH_MOVE:
		m_stage = QUEEN_PROMOTIONS;

		break;
	case QUEEN_PROMOTIONS:
		m_stage = WINNING_EQUAL_CAPTURES;
		break;
	case WINNING_EQUAL_CAPTURES:
		if (m_isQS)
		{
			// if we are in QS, this is it, don't advance to next stage
		}
		else
		{
			m_stage = KILLERS;
		}
		break;
	case KILLERS:
		m_stage = OTHER_NON_CAPTURES;

		break;
	case OTHER_NON_CAPTURES:
		m_stage = LOSING_CAPTURES;
		break;
	case LOSING_CAPTURES:
		// nothing to do here
		break;
	default:
		assert(false);
	}
}

void MovePicker::AssignSeeScores_(MoveList &ml)
{
	for (size_t i = 0; i < ml.GetSize(); ++i)
	{
		Score seeScore = SEE::StaticExchangeEvaluation(m_board, ml[i]);
		SetScore(ml[i], seeScore + 0x8000);
	}
}

void MovePicker::RemoveScores_(MoveList &ml)
{
	for (size_t i = 0; i < ml.GetSize(); ++i)
	{
		ClearScore(ml[i]);
	}
}

void DebugMovePicker(Board &b, uint32_t depth, Killer &killer)
{
	MoveList ml;
	b.GenerateAllLegalMoves<Board::ALL>(ml);

	MoveList mlq;
	b.GenerateAllLegalMoves<Board::VIOLENT>(mlq);

	MoveList mlqu;
	b.GenerateAllLegalMoves<Board::QUIET>(mlqu);

	assert((mlq.GetSize() + mlqu.GetSize()) == ml.GetSize());

	MoveList returnedMoves;

	size_t equalWinningCaptures = 0;

	for (size_t i = 0; i < mlq.GetSize(); ++i)
	{
		if (SEE::StaticExchangeEvaluation(b, mlq[i]) >= 0)
		{
			++equalWinningCaptures;
		}
	}

	Board c = b;

	Move hashMove = 0;
	if (b.GetHash() & 0x100ULL && ml.GetSize() > 5)
	{
		hashMove = ml[5];
	}

	if (b.GetHash() & 0x1000ULL)
	{
		killer.Notify(b.GetSideToMove() == WHITE ? 4 : 5, ml[b.GetHash() % ml.GetSize()]);
	}

	if (b.GetHash() & 0x10000ULL)
	{
		killer.Notify(b.GetSideToMove() == WHITE ? 2 : 3, ml[b.GetHash() % ml.GetSize()]);
	}

	MovePicker mp(b, hashMove, killer, false, b.GetSideToMove() == WHITE ? 4 : 5);

	Move mv;

	while ((mv = mp.GetNextMove()))
	{
		if (!b.CheckPseudoLegal(mv))
		{
			std::cout << "Move picker returned an illegal move!" << std::endl;
			std::cout << b.MoveToAlg(mv) << std::endl;
			abort();
		}

		if (b.GetHash() % 0x100ULL && ml.GetSize() > 5)
		{
			if (returnedMoves.GetSize() == 0)
			{
				if (hashMove && ClearScore(mv) != hashMove)
				{
					std::cout << "First move returned is not hash move!" << std::endl;
					std::cout << "FEN: " << b.GetFen() << std::endl;
					std::cout << "Hash move: " << b.MoveToAlg(hashMove) << std::endl;
					std::cout << "Returned: " << b.MoveToAlg(mv) << std::endl;
					abort();
				}
			}
			else
			{
				if (ClearScore(mv) == hashMove)
				{
					std::cout << "Hash move returned twice!" << std::endl;
					abort();
				}
			}
		}

		returnedMoves.PushBack(mv);

		if (b.ApplyMove(mv))
		{
			if (depth != 1)
			{
				DebugMovePicker(b, depth - 1, killer);
			}

			b.UndoMove();

			assert(b == c);
		}
	}

	if (returnedMoves.GetSize() != ml.GetSize())
	{
		std::cout << "Move picker size mismatch! Expected " << ml.GetSize() << " Got " << returnedMoves.GetSize() << std::endl;
		std::cout << b.PrintBoard() << std::endl;
		std::cout << "Expected:" << std::endl;
		for (size_t i = 0; i < ml.GetSize(); ++i)
		{
			std::cout << b.MoveToAlg(ml[i]) << std::endl;
		}

		std::cout << "Got:" << std::endl;
		for (size_t i = 0; i < returnedMoves.GetSize(); ++i)
		{
			std::cout << b.MoveToAlg(returnedMoves[i]) << std::endl;
		}
		abort();
	}

	MovePicker mpq(b, 0, killer, true, b.GetSideToMove() == WHITE ? 4 : 5);
	returnedMoves.Clear();
	mv = 0;

	while ((mv = mpq.GetNextMove()))
	{
		if (!b.CheckPseudoLegal(mv))
		{
			std::cout << "Move picker returned an illegal move!" << std::endl;
			std::cout << b.MoveToAlg(mv) << std::endl;
			abort();
		}

		returnedMoves.PushBack(mv);

		if (b.ApplyMove(mv))
		{
			if (depth != 1)
			{
				DebugMovePicker(b, depth - 1, killer);
			}

			b.UndoMove();

			assert(b == c);
		}
	}

	if (returnedMoves.GetSize() != equalWinningCaptures)
	{
		std::cout << "Move picker size mismatch for QS! Expected " << equalWinningCaptures << " Got " << returnedMoves.GetSize() << std::endl;
		std::cout << b.PrintBoard() << std::endl;

		std::cout << "Expected:" << std::endl;
		for (size_t i = 0; i < mlq.GetSize(); ++i)
		{
			if (SEE::StaticExchangeEvaluation(b, mlq[i]) >= 0)
			{
				std::cout << b.MoveToAlg(mlq[i]) << std::endl;
			}
		}

		std::cout << "Got:" << std::endl;
		for (size_t i = 0; i < returnedMoves.GetSize(); ++i)
		{
			std::cout << b.MoveToAlg(returnedMoves[i]) << std::endl;
		}

		abort();
	}
}

void CheckMovePicker(std::string fen, uint32_t depth)
{
	std::cout << "Checking move picker for " << fen << ", Depth: " << depth << std::endl;
	Board b(fen);
	Killer killer;
	DebugMovePicker(b, depth, killer);
}

void DebugRunMovePickerTests()
{
	CheckMovePicker("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4);
	CheckMovePicker("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 3);
	CheckMovePicker("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 5);
	CheckMovePicker("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4);
	CheckMovePicker("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 4);
	CheckMovePicker("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6", 1);
	CheckMovePicker("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3);

	std::cout << "Checking special case - " << "2r4k/1P6/8/4q1nr/7p/5N2/K7/8 w - - 0 1" << std::endl;
	Board b("2r4k/1P6/8/4q1nr/7p/5N2/K7/8 w - - 0 1");
	Killer killer;

	killer.Notify(4, b.ParseMove("f3h4")); // this is a violent move at this ply, so should be filtered out
	killer.Notify(4, b.ParseMove("f3g1")); // this is a good killer

	Move hashMove = b.ParseMove("b7c8r"); // capturing under-promotion

	MovePicker mp(b, hashMove, killer, false, 4);

	Move mv;
	// first we should get the hash move
	mv = mp.GetNextMove(); if (ClearScore(mv) != b.ParseMove("b7c8r")) { std::cout << "Got " << b.MoveToAlg(mv) << std::endl; abort(); }

	// then we should get the capturing queen promotions, since the non-capturing one loses the queen right away (SEE)
	mv = mp.GetNextMove(); if (ClearScore(mv) != b.ParseMove("b7c8q")) { std::cout << "Got " << b.MoveToAlg(mv) << std::endl; abort(); }

	// then we should get the winning and equal captures (except for the pawn promotion, because that's generated already)
	mv = mp.GetNextMove(); if (ClearScore(mv) != b.ParseMove("f3e5")) { std::cout << "Got " << b.MoveToAlg(mv) << std::endl; abort(); }
	mv = mp.GetNextMove(); if (ClearScore(mv) != b.ParseMove("f3g5")) { std::cout << "Got " << b.MoveToAlg(mv) << std::endl; abort(); }

	// then we should get the valid killer only
	mv = mp.GetNextMove(); if (ClearScore(mv) != b.ParseMove("f3g1")) { std::cout << "Got " << b.MoveToAlg(mv) << std::endl; abort(); }

	// then we should get neutral moves
	// f3d2, a2b1, a2a3, a2b3
	MoveList neutralMovesExpected;
	neutralMovesExpected.PushBack(b.ParseMove("f3d2"));
	neutralMovesExpected.PushBack(b.ParseMove("a2b1"));
	neutralMovesExpected.PushBack(b.ParseMove("a2a3"));
	neutralMovesExpected.PushBack(b.ParseMove("a2b3"));

	MoveList neutralMovesReturned;
	for (size_t i = 0; i < 4; ++i)
	{
		Move mv = ClearScore(mp.GetNextMove());
		neutralMovesReturned.PushBack(mv);
	}

	if (!neutralMovesExpected.CompareUnorderedSlow(neutralMovesReturned))
	{
		std::cout << "Neutral moves returned incorrect" << std::endl;
		abort();
	}

	size_t losingMovesCount = 0;
	while ((mv = mp.GetNextMove()))
	{
		++losingMovesCount;
	}

	// there should be 12 losing moves
	// b7b8q, f3h4, b7b8r, b7b8n, b7b8b, b7c8n, b7c8b, f3e1, f3h2, f3d4, a2a1, a2b2
	if (losingMovesCount != 12)
	{
		std::cout << "Losing moves count wrong: returned " << losingMovesCount << std::endl;
		abort();
	}

	std::cout << "Done" << std::endl;
}

