#ifndef BACKEND_H
#define BACKEND_H

#include <memory>
#include <mutex>

#include "board.h"
#include "search.h"

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

	void DebugPrintBoard();

private:
	void StopSearch_();
	void StartSearch_(double timeAllocated, double maxTimeAllocated, Search::SearchType searchType);

	std::mutex m_mutex;

	EngineMode m_mode;
	Board m_currentBoard;
	bool m_searchInProgress;
	std::unique_ptr<Search::AsyncSearch> m_search;
	std::unique_ptr<Search::RootSearchContext> m_searchContext;

	Search::Depth m_maxDepth;

	bool m_showThinking;
};

#endif // BACKEND_H
