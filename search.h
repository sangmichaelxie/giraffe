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

#ifndef SEARCH_H
#define SEARCH_H

#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <future>
#include <functional>
#include <condition_variable>

#include <cmath>

#include "types.h"
#include "board.h"
#include "countermove.h"
#include "history.h"
#include "ttable.h"
#include "eval/eval.h"
#include "killer.h"
#include "evaluator.h"
#include "move_evaluator.h"

namespace Search
{

typedef int32_t Depth;

// this function is for converting CEPT/UCI depth settings to some estimate of node budget
inline NodeBudget DepthToNodeBudget(Depth d)
{
	return std::pow(4, d);
}

static const bool ENABLE_NULL_MOVE_HEURISTICS = true;

static const NodeBudget MinNodeBudgetForNullMove = 1;
static const float NullMoveNodeBudgetMultiplier = 0.0003f;

static const bool ENABLE_TT = true;

static const bool ENABLE_IID = true;
static const NodeBudget MinNodeBudgetForIID = 1024;
static const float IIDNodeBudgetMultiplier = 0.1f;

static const bool ENABLE_PVS = true;
static const NodeBudget MinNodeBudgetForPVS = 16;

static const bool ENABLE_KILLERS = true;

static const bool ENABLE_COUNTERMOVES = false;

static const bool ENABLE_HISTORY = true;

static const Score ASPIRATION_WINDOW_HALF_SIZE = 400;

// if we get above this size, just open wide
// this prevents many researches when a mate score is first discovered
// this must be less than max value for the type divided by WIDEN_MULTIPLIER, otherwise there is a potential for overflow
static const Score ASPIRATION_WINDOW_HALF_SIZE_THRESHOLD = 1600;

static const Score ASPIRATION_WINDOW_WIDEN_MULTIPLIER = 4; // how much to widen the window every time we fail high/low

static const Score DRAW_SCORE = 0;
static const size_t NUM_MOVES_TO_LOOK_FOR_DRAW = 16; // how many moves past to look for draws (we are only looking for 2-fold)

struct ThinkingOutput
{
	int32_t ply;
	Score score;
	double time;
	uint64_t nodeCount;
	std::string pv;
};

struct TimeAllocation
{
	double normalTime; // time allocated for this move if nothing special happens
	double maxTime; // absolute maximum time for this move
};

struct SearchResult
{
	Score score;

	std::vector<Move> pv;
};

enum SearchType
{
	SearchType_makeMove, // search using the time allocated, and make a move in the end
	SearchType_infinite // search until told to stop (ponder, analyze)
};

// all searches starting from the same root will have the same context
// must be thread-safe
struct RootSearchContext
{
	// time allocated for this root search
	TimeAllocation timeAlloc;

	std::atomic<bool> onePlyDone;
	std::atomic<bool> stopRequest;

	Board startBoard;

	std::atomic<uint64_t> nodeCount;

	SearchType searchType;

	NodeBudget nodeBudget;

	TTable *transpositionTable;
	Killer *killer;
	CounterMove *counter;
	History *history;

	EvaluatorIface *evaluator;
	MoveEvaluatorIface *moveEvaluator;

	std::function<void (std::string &mv)> finalMoveFunc;
	std::function<void (ThinkingOutput &to)> thinkingOutputFunc;

	bool Stopping() { return onePlyDone && stopRequest; }
};

class AsyncSearch
{
public:
	AsyncSearch(RootSearchContext &context);

	void Start();

	// request an abort (must wait for search to be done)
	void Abort()
	{ std::lock_guard<std::mutex> lock(m_abortingMutex); m_context.stopRequest = true; m_cvAborting.notify_all(); }

	// whether the search is done (or aborted)
	bool Done() { return m_done; }

	void Join() { if (m_thread.joinable()) { m_thread.join(); } }

	// result is only defined once search is done
	SearchResult GetResult() { return m_rootResult; }

private:
	void RootSearch_();

	// entry point for a thread that automatically interrupts the search after the specified time
	void SearchTimer_(double time);

	RootSearchContext &m_context;
	std::thread m_thread;
	std::atomic<bool> m_done;

	SearchResult m_rootResult;

	std::mutex m_abortingMutex;
	std::condition_variable m_cvAborting;

	std::thread m_searchTimerThread;
};

Score Search(RootSearchContext &context, std::vector<Move> &pv, Board &board, Score alpha, Score beta, NodeBudget nodeBudget, int32_t ply, bool nullMoveAllowed = true);

Score QSearch(RootSearchContext &context, std::vector<Move> &pv, Board &board, Score alpha, Score beta, int32_t ply, int32_t qsPly);

// perform a synchronous search (no thread creation)
// this is used in training only, where we don't want to do a typical root search, and don't want all the overhead
SearchResult SyncSearchNodeLimited(const Board &b, NodeBudget nodeBudget, EvaluatorIface *evaluator, MoveEvaluatorIface *moveEvaluator, Killer *killer = nullptr, TTable *ttable = nullptr, CounterMove *counter = nullptr, History *history = nullptr);

// print search trees for debugging
extern bool trace;

}

#endif // SEARCH_H
