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

namespace Search
{

typedef int32_t Depth;

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

	Score Search_(RootSearchContext &context, Move &bestMove, Board &board, Score alpha, Score beta, Depth depth, int32_t ply);
	Score Search_(RootSearchContext &context, Board &board, Score alpha, Score beta, Depth depth, int32_t ply); // version without best move

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
