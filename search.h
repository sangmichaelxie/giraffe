#ifndef SEARCH_H
#define SEARCH_H

#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <future>

#include "types.h"
#include "board.h"
#include "eval/eval.h"

namespace Search
{

typedef int32_t Depth;

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

// all searches starting from the same root will have the same context
// must be thread-safe
struct RootSearchContext
{
	// time allocated for this root search
	TimeAllocation timeAlloc;

	std::atomic<bool> stopRequest;

	Board startBoard;

	std::atomic<uint64_t> nodeCount;
};

class AsyncSearch
{
public:
	AsyncSearch(RootSearchContext &context);

	void Start();

	// request an abort (must wait for search to be done)
	void Abort() { m_context.stopRequest = true; }

	// whether the search is done (or aborted)
	bool Done() { return m_done; }

	void Join() { if (m_thread.joinable()) { m_thread.join(); } }

	// result is only defined once search is done
	SearchResult GetResult() { return m_rootResult; }

private:
	void RootSearch_();

	Score Search_(RootSearchContext &context, Move &bestMove, Board &board, Score alpha, Score beta, Depth depth, int32_t ply);
	Score Search_(RootSearchContext &context, Board &board, Score alpha, Score beta, Depth depth, int32_t ply); // version without best move

	RootSearchContext &m_context;
	std::thread m_thread;
	std::atomic<bool> m_done;

	SearchResult m_rootResult;
};

}

#endif // SEARCH_H
