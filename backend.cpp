#include "backend.h"

#include "board.h"

Backend::Backend()
	: m_mode(Backend::EngineMode_force), m_searchInProgress(false), m_showThinking(false)
{
}

Backend::~Backend()
{
	StopSearch_();
}

void Backend::NewGame()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	StopSearch_();

	m_mode = EngineMode_force;

	m_currentBoard = Board();
}

void Backend::Force()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	StopSearch_();

	m_mode = EngineMode_force;
}

void Backend::Go()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	StopSearch_();

	if (m_currentBoard.GetSideToMove() == WHITE)
	{
		m_mode = EngineMode_playingWhite;
	}
	else
	{
		m_mode = EngineMode_playingBlack;
	}

	StartSearch_(10.0, 20.0, Search::SearchType_makeMove);
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

	m_currentBoard.ApplyMove(parsedMove);

	if (m_mode == EngineMode_playingWhite || m_mode == EngineMode_playingBlack)
	{
		StopSearch_();
		StartSearch_(10.0, 20.0, Search::SearchType_makeMove);
	}
	else if (m_mode == EngineMode_analyzing)
	{
		StopSearch_();
		StartSearch_(0.0, 0.0, Search::SearchType_infinite);
	}
}

void Backend::SetBoard(std::string fen)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	StopSearch_();

	m_mode = EngineMode_force;

	m_currentBoard = Board(fen);
}

void Backend::SetAnalyzing(bool enabled)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	if (enabled)
	{
		m_mode = EngineMode_analyzing;

		StartSearch_(0.0, 0.0, Search::SearchType_infinite);
	}
	else
	{
		StopSearch_();

		m_mode = EngineMode_force;
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
		StopSearch_();
		StartSearch_(0.0, 0.0, Search::SearchType_infinite);
	}
}

void Backend::DebugPrintBoard()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	std::cout << m_currentBoard.PrintBoard() << std::endl;
}

void Backend::StopSearch_()
{
	if (m_searchInProgress)
	{
		m_search->Abort();
		m_search->Join();

		m_searchInProgress = false;
	}
}

void Backend::StartSearch_(double timeAllocated, double maxTimeAllocated, Search::SearchType searchType)
{
	m_searchInProgress = true;

	m_searchContext.reset(new Search::RootSearchContext());

	m_searchContext->timeAlloc.normalTime = timeAllocated;
	m_searchContext->timeAlloc.maxTime = maxTimeAllocated;
	m_searchContext->stopRequest = false;
	m_searchContext->startBoard = m_currentBoard;
	m_searchContext->nodeCount = 0;
	m_searchContext->searchType = searchType;

	m_searchContext->thinkingOutputFunc =
	[this](Search::ThinkingOutput &to)
	{
		std::lock_guard<std::mutex> lock(m_mutex);

		if (m_showThinking)
		{
			std::cout << to.ply << " " << to.score << " " << static_cast<int64_t>(to.time * 100.0) <<
						 " " << to.nodeCount << " " << to.pv << std::endl;
		}
	};

	m_searchContext->finalMoveFunc =
	[this](std::string &mv)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		std::cout << "move " << mv << std::endl;
		m_currentBoard.ApplyMove(m_currentBoard.ParseMove(mv));
	};

	m_search.reset(new Search::AsyncSearch(*m_searchContext));

	m_search->Start();
}
