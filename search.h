#ifndef SEARCH_H
#define SEARCH_H

#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <future>
#include <functional>
#include <condition_variable>

#include "types.h"
#include "board.h"
#include "ttable.h"
#include "eval/eval.h"
#include "killer.h"

namespace Search
{

typedef int32_t Depth;

static const bool ENABLE_NULL_MOVE_HEURISTICS = true;
static const bool ENABLE_ADAPTIVE_NULL_MOVE = false;

static const bool NM_REDUCE_INSTEAD_OF_PRUNE = false;

static const bool ENABLE_IID = true;
static const bool ENABLE_PVS = true;
static const bool ENABLE_KILLERS = true;
static const bool ENABLE_FUTILITY_PRUNING = true;
static const bool ENABLE_BAD_MOVE_PRUNING = false;

static const bool ENABLE_LATE_MOVE_REDUCTION = false;

// besides late move reduction, we can further reduce bad moves (moves that leave pieces hanging, but not losing captures)
static const bool ENABLE_BAD_MOVE_REDUCTION = false;

static const Depth NULL_MOVE_REDUCTION = 3;
static const int32_t ADAPTIVE_NULL_MOVE_THRESHOLD = 6;

// how many plies to reduce if a null move fails high (this is only used if NM_REDUCE_INSTEAD_OF_PRUNE is set)
static const Depth NMR_DR = 4;

static const Score ASPIRATION_WINDOW_HALF_SIZE = 25;
static const Depth LMR_MIN_DEPTH = 3;

static const int32_t LMR_NUM_MOVES_FULL_DEPTH = 2;
static const Depth LATE_MOVE_REDUCTION = 1;

static const int32_t LMR_NUM_MOVES_REDUCE_1 = 6; // additional reduction after this many moves

static const Depth BAD_MOVE_REDUCTION = 1; // this is in addition to regular LMR
static const Depth BAD_MOVE_PRUNING_MAX_DEPTH = 3;

// if we get above this size, just open wide
// this prevents many researches when a mate score is first discovered
// this must be less than max value for the type divided by WIDEN_MULTIPLIER, otherwise there is a potential for overflow
static const Score ASPIRATION_WINDOW_HALF_SIZE_THRESHOLD = 500;

static const Score ASPIRATION_WINDOW_WIDEN_MULTIPLIER = 4; // how much to widen the window every time we fail high/low

// futility thresholds indexed by parent remaining depth
// these values are from Crafty
//static const Score FUTILITY_MARGINS[] = { 0, 100, 100, 200, 200, 300, 300, 400 };
static const Score FUTILITY_MARGINS[] = { 0, 200, 500 };
static const int32_t FUTILITY_MAX_DEPTH = 3; // this is actually 1 + max depth

static const Score DRAW_SCORE = 0;
static const size_t NUM_MOVES_TO_LOOK_FOR_DRAW = 8; // how many moves past to look for draws (we are only looking for 2-fold)

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
	Move bestMove;
	Score score;
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

	Depth maxDepth;

	TTable *transpositionTable;
	Killer *killer;

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

	Score Search_(RootSearchContext &context, Move &bestMove, Board &board, Score alpha, Score beta, Depth depth, int32_t ply, bool nullMoveAllowed = true);
	Score Search_(RootSearchContext &context, Board &board, Score alpha, Score beta, Depth depth, int32_t ply, bool nullMoveAllowed = true); // version without best move

	Score QSearch_(RootSearchContext &context, Board &board, Score alpha, Score beta, int32_t ply);

	RootSearchContext &m_context;
	std::thread m_thread;
	std::atomic<bool> m_done;

	SearchResult m_rootResult;

	std::mutex m_abortingMutex;
	std::condition_variable m_cvAborting;

	std::thread m_searchTimerThread;
};

}

#endif // SEARCH_H
