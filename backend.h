#ifndef BACKEND_H
#define BACKEND_H

#include <memory>
#include <mutex>

#include "board.h"
#include "search.h"
#include "chessclock.h"

class Backend
{
public:
	enum EngineMode
	{
		EngineMode_force,
		EngineMode_playingWhite,
		EngineMode_playingBlack,
		EngineMode_analyzing // analyzing is the same as force, except we think
	};

	Backend();
	~Backend();

	void NewGame();

	void Force();

	void Go();

	void Usermove(std::string move);

	void SetBoard(std::string fen);

	void SetShowThinking(bool enabled) { std::lock_guard<std::mutex> lock(m_mutex); m_showThinking = enabled; }

	void SetMaxDepth(Search::Depth depth) { std::lock_guard<std::mutex> lock(m_mutex); m_maxDepth = depth; }

	void SetAnalyzing(bool enabled);

	void Undo(int32_t moves);

	void SetTimeControl(const ChessClock &cc)
	{ std::lock_guard<std::mutex> lock(m_mutex); m_whiteClock = cc; m_blackClock = cc; }

	void AdjustEngineTime(double time);
	void AdjustOpponentTime(double time);

	void DebugPrintBoard();
	void DebugRunPerft(int32_t depth);
	Score DebugEval();
	void DebugPerftTests() { DebugRunPerftTests(); }

private:
	// these 2 functions take a lock_guard to remind the caller that m_mutex should be locked when calling
	// these functions, since these functions will temporarily unlock the mutex while waiting for search
	// thread to join
	void Force_(std::lock_guard<std::mutex> &lock);
	void StopSearch_(std::lock_guard<std::mutex> &lock);

	void StartSearch_(Search::SearchType searchType);

	std::mutex m_mutex;

	EngineMode m_mode;
	Board m_currentBoard;
	bool m_searchInProgress;
	std::unique_ptr<Search::AsyncSearch> m_search;
	std::unique_ptr<Search::RootSearchContext> m_searchContext;

	Search::Depth m_maxDepth;

	bool m_showThinking;

	ChessClock m_whiteClock;
	ChessClock m_blackClock;
};

#endif // BACKEND_H
