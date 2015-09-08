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

#include "backend.h"

#include <iostream>
#include <string>

#include "board.h"
#include "timeallocator.h"
#include "eval/eval.h"
#include "gtb.h"

Backend::Backend()
	: m_mode(Backend::EngineMode_force), m_searchInProgress(false), m_maxDepth(0), m_showThinking(false),
	  m_whiteClock(ChessClock::CONVENTIONAL_INCREMENTAL_MODE, 0, 300, 0),
	  m_blackClock(ChessClock::CONVENTIONAL_INCREMENTAL_MODE, 0, 300, 0),
	  m_tTable(DEFAULT_TTABLE_SIZE / sizeof(TTEntry)),
	  m_evaluator(&Eval::gStaticEvaluator),
	  m_moveEvaluator(&gStaticMoveEvaluator)
{
}

Backend::~Backend()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	StopSearch_(lock);
}

void Backend::NewGame()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	Force_(lock);

	m_currentBoard = Board();

	m_tTable.ClearTable();

	m_mode = EngineMode_playingBlack;
}

void Backend::Force()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	Force_(lock);
}

void Backend::Go()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	StopSearch_(lock);

	if (m_currentBoard.GetSideToMove() == WHITE)
	{
		m_mode = EngineMode_playingWhite;
	}
	else
	{
		m_mode = EngineMode_playingBlack;
	}

	StartSearch_(Search::SearchType_makeMove);
}

void Backend::Usermove(std::string move)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	Move parsedMove = m_currentBoard.ParseMove(move);

	if (!parsedMove)
	{
		std::cout << "Illegal move: " << move << std::endl;
		return;
	}

	if ((m_mode == EngineMode_playingWhite && m_currentBoard.GetSideToMove() == WHITE) ||
		(m_mode == EngineMode_playingBlack && m_currentBoard.GetSideToMove() == BLACK))
	{
		std::cout << "Illegal move (out of turn): " << move << std::endl;
		return;
	}

	StopSearch_(lock);

	m_currentBoard.ApplyMove(parsedMove);

	if (!CheckDeclareGameResult_())
	{
		m_mode = EngineMode_force;
		return;
	}

	if (m_mode == EngineMode_playingWhite || m_mode == EngineMode_playingBlack)
	{
		StartSearch_(Search::SearchType_makeMove);

		if (m_mode == EngineMode_playingWhite)
		{
			m_blackClock.Stop();
			m_whiteClock.Start();
		}
		else
		{
			m_whiteClock.Stop();
			m_blackClock.Start();
		}
	}
	else if (m_mode == EngineMode_analyzing)
	{
		StartSearch_(Search::SearchType_infinite);
	}

	m_tTable.AgeTable();
	m_killer.MoveMade();
	m_history.NotifyMoveMade();
}

void Backend::SetBoard(std::string fen)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	Force_(lock);

	m_currentBoard = Board(fen);

	m_tTable.ClearTable();
}

void Backend::SetAnalyzing(bool enabled)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	Force_(lock);

	if (enabled)
	{
		m_mode = EngineMode_analyzing;

		StartSearch_(Search::SearchType_infinite);
	}
}

void Backend::Undo(int32_t moves)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	if (m_currentBoard.PossibleUndo() < moves)
	{
		std::cout << "Error (no moves to undo)" << std::endl;
		return;
	}

	for (int32_t i = 0; i < moves; ++i)
	{
		m_currentBoard.UndoMove();
	}

	// we will only be sent "remove" (undo 2 moves) if it's user's move, and
	// "undo" (undo 1 move) in force or analyze mode, so we will never have to start
	// thinking to make a move after undo

	if (m_mode == EngineMode_analyzing)
	{
		StopSearch_(lock);
		StartSearch_(Search::SearchType_infinite);
	}
}

void Backend::AdjustEngineTime(double time)
{
	if (m_mode == EngineMode_playingWhite)
	{
		m_whiteClock.AdjustTime(time);
	}
	else if (m_mode == EngineMode_playingBlack)
	{
		m_blackClock.AdjustTime(time);
	}
	else
	{
		std::cout << "Error (not playing a game)" << std::endl;
	}
}

void Backend::AdjustOpponentTime(double time)
{
	if (m_mode == EngineMode_playingWhite)
	{
		m_blackClock.AdjustTime(time);
	}
	else if (m_mode == EngineMode_playingBlack)
	{
		m_whiteClock.AdjustTime(time);
	}
	else
	{
		std::cout << "Error (not playing a game)" << std::endl;
	}
}

void Backend::DebugPrintBoard()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	std::cout << m_currentBoard.PrintBoard() << std::endl;
}

void Backend::DebugRunPerft(int32_t depth)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	DebugPerft(m_currentBoard, depth);
}

void Backend::DebugRunPerftWithNull(int32_t depth)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	DebugPerftWithNull(m_currentBoard, depth);
}

Score Backend::DebugEval()
{
	return m_evaluator->EvaluateForSTM(m_currentBoard, SCORE_MIN, SCORE_MAX);
}

void Backend::PrintDebugEval()
{
	m_evaluator->PrintDiag(m_currentBoard);
}

void Backend::PrintDebugMoveEval()
{
	m_moveEvaluator->PrintDiag(m_currentBoard);
}

std::string Backend::DebugGTB()
{
	GTB::ProbeResult result = GTB::Probe(m_currentBoard);

	if (!result)
	{
		return "No result";
	}
	else
	{
		return ToStr(*result);
	}
}

void Backend::Quit()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	StopSearch_(lock);
}

bool Backend::IsAMove(const std::string &s)
{
	Move parsedMove = m_currentBoard.ParseMove(s);

	return (parsedMove != 0);
}

void Backend::Force_(std::lock_guard<std::mutex> &lock)
{
	StopSearch_(lock);

	m_whiteClock.Stop();
	m_blackClock.Stop();

	m_mode = EngineMode_force;
}

void Backend::StopSearch_(std::lock_guard<std::mutex> &/*lock*/)
{
	if (m_searchInProgress)
	{
		m_search->Abort();

		m_mutex.unlock();
		try
		{
			m_search->Join();
		}
		catch (...)
		{}
		m_mutex.lock();

		m_searchInProgress = false;
	}
}

void Backend::StartSearch_(Search::SearchType searchType)
{
	m_searchInProgress = true;

	m_searchContext.reset(new Search::RootSearchContext());

	Search::TimeAllocation tAlloc;

	if (m_mode == EngineMode_playingWhite)
	{
		tAlloc = AllocateTime(m_whiteClock);
	}
	else if (m_mode == EngineMode_playingBlack)
	{
		tAlloc = AllocateTime(m_blackClock);
	}

	m_searchContext->timeAlloc = tAlloc;
	m_searchContext->onePlyDone = false;
	m_searchContext->stopRequest = false;
	m_searchContext->startBoard = m_currentBoard;
	m_searchContext->nodeCount = 0;
	m_searchContext->searchType = searchType;
	m_searchContext->nodeBudget = m_maxDepth == 0 ? 0 : Search::DepthToNodeBudget(m_maxDepth);
	m_searchContext->transpositionTable = &m_tTable;
	m_searchContext->killer = &m_killer;
	m_searchContext->counter = &m_counter;
	m_searchContext->history = &m_history;

	m_searchContext->evaluator = m_evaluator;
	m_searchContext->moveEvaluator = m_moveEvaluator;

	m_searchContext->thinkingOutputFunc =
	[this](Search::ThinkingOutput &to)
	{
		std::lock_guard<std::mutex> lock(m_mutex);

		const float OutputScoreScale = 0.1f;

		if (m_showThinking)
		{
			std::cout << to.ply << " " << static_cast<int64_t>(OutputScoreScale * to.score) << " " << static_cast<int64_t>(to.time * 100.0) <<
						 " " << to.nodeCount << " " << to.pv << std::endl;
		}
	};

	m_searchContext->finalMoveFunc =
	[this](std::string &mv)
	{
		std::lock_guard<std::mutex> lock(m_mutex);

		m_currentBoard.ApplyMove(m_currentBoard.ParseMove(mv));

		// if we want to claim a draw, we have to send it before sending the move
		if (m_currentBoard.Is3Fold() || m_currentBoard.Is50Moves())
		{
			// here we use the offer draw command instead of claiming a result
			// it's safer this way because if the GUI doesn't agree this is a draw
			// we can simply play on
			std::cout << "offer draw" << std::endl;
		}

		std::cout << "move " << mv << std::endl;

		if (!CheckDeclareGameResult_())
		{
			m_mode = EngineMode_force;
			return;
		}

		m_tTable.AgeTable();
		m_killer.MoveMade();
		m_history.NotifyMoveMade();

		if (m_mode == EngineMode_playingBlack)
		{
			m_blackClock.Stop();
			m_whiteClock.Start();
		}
		else
		{
			m_whiteClock.Stop();
			m_blackClock.Start();
		}
	};

	m_search.reset(new Search::AsyncSearch(*m_searchContext));

	m_search->Start();
}

bool Backend::CheckDeclareGameResult_()
{
	Board::GameStatus gameResult = m_currentBoard.GetGameStatus();

	if (gameResult == Board::ONGOING)
	{
		return true;
	}

	if (gameResult == Board::WHITE_WINS)
	{
		std::cout << "1-0 {White mates}" << std::endl;
	}
	else if (gameResult == Board::BLACK_WINS)
	{
		std::cout << "0-1 {Black mates}" << std::endl;
	}
	else if (gameResult == Board::STALEMATE)
	{
		std::cout << "1/2-1/2 {Stalemate}" << std::endl;
	}
	else if (gameResult == Board::INSUFFICIENT_MATERIAL)
	{
		std::cout << "1/2-1/2 {Draw by insufficient material}" << std::endl;
	}

	return false;
}
