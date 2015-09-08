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

#ifndef BACKEND_H
#define BACKEND_H

#include <memory>
#include <mutex>

#include "board.h"
#include "search.h"
#include "chessclock.h"
#include "history.h"
#include "ttable.h"
#include "killer.h"
#include "move_evaluator.h"
#include "static_move_evaluator.h"
#include "countermove.h"

class Backend
{
public:
	const static size_t DEFAULT_TTABLE_SIZE = 256*MB; // 256MB

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

	void SetEvaluator(EvaluatorIface *newEvaluator) { m_evaluator = newEvaluator; }

	EvaluatorIface *GetEvaluator() { return m_evaluator; }

	void SetMoveEvaluator(MoveEvaluatorIface *newMoveEvaluator) { m_moveEvaluator = newMoveEvaluator; }

	MoveEvaluatorIface *GetMoveEvaluator() { return m_moveEvaluator; }

	void DebugPrintBoard();
	void DebugRunPerft(int32_t depth);
	void DebugRunPerftWithNull(int32_t depth);
	Score DebugEval();
	void PrintDebugEval();

	void PrintDebugMoveEval();

	std::string DebugGTB();

	void Quit();

	bool IsAMove(const std::string &s);

	Board &GetBoard() { return m_currentBoard; }

private:
	// these 2 functions take a lock_guard to remind the caller that m_mutex should be locked when calling
	// these functions, since these functions will temporarily unlock the mutex while waiting for search
	// thread to join
	void Force_(std::lock_guard<std::mutex> &lock);
	void StopSearch_(std::lock_guard<std::mutex> &lock);

	void StartSearch_(Search::SearchType searchType);

	// returns whether the game is still ongoing
	bool CheckDeclareGameResult_();

	std::mutex m_mutex;

	EngineMode m_mode;
	Board m_currentBoard;
	bool m_searchInProgress;
	std::unique_ptr<Search::AsyncSearch> m_search;
	std::unique_ptr<Search::RootSearchContext> m_searchContext;

	// this is the max depth set by the protocol
	// we aren't doing depth limited search, so we have to convert it to node budget
	// when we actually do a search
	Search::Depth m_maxDepth;

	bool m_showThinking;

	ChessClock m_whiteClock;
	ChessClock m_blackClock;

	TTable m_tTable;
	Killer m_killer;
	CounterMove m_counter;
	History m_history;

	EvaluatorIface *m_evaluator;
	MoveEvaluatorIface *m_moveEvaluator;
};

#endif // BACKEND_H
