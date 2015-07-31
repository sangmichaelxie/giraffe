#include "search.h"

#include <utility>
#include <memory>
#include <atomic>
#include <chrono>

#include <cstdint>

#include "types.h"
#include "util.h"
#include "eval/eval.h"
#include "see.h"
#include "gtb.h"

namespace
{
	// estimated minimum branching factor for time allocation
	// if more than 1/x of the allocated time has been used at the end of an iteration,
	// a new iteration won't be started
	const static double ESTIMATED_MIN_BRANCHING_FACTOR = 5.0;
}

namespace Search
{

static const Depth ID_MAX_DEPTH = 200;

AsyncSearch::AsyncSearch(RootSearchContext &context)
	: m_context(context), m_done(false)
{
}

void AsyncSearch::Start()
{
	std::thread searchThread(&AsyncSearch::RootSearch_, this);

	m_thread = std::move(searchThread);
}

void AsyncSearch::RootSearch_()
{
	double startTime = CurrentTime();

	// this is the time we want to stop searching
	double normaleEndTime = startTime + m_context.timeAlloc.normalTime;

	// this is the time we HAVE to stop searching
	// only the ID controller can make the decision to switch to this end time
	//double absoluteEndTime = startTime + m_context.timeAlloc.maxTime;

	double endTime = normaleEndTime;

	SearchResult latestResult;

	if (m_context.searchType != SearchType_infinite)
	{
		std::thread searchTimerThread(&AsyncSearch::SearchTimer_, this, endTime - CurrentTime());
		m_searchTimerThread = std::move(searchTimerThread);
	}

	if (m_context.maxDepth == 0 || m_context.maxDepth > ID_MAX_DEPTH)
	{
		m_context.maxDepth = ID_MAX_DEPTH;
	}

	m_context.onePlyDone = false;

	latestResult.score = 0;

	for (Depth depth = 1;
			(depth <= m_context.maxDepth) &&
			((CurrentTime() < endTime) || (m_context.searchType == SearchType_infinite) || !m_context.onePlyDone) &&
			(!m_context.Stopping());
		 ++depth)
	{
		// aspiration search
		Score lastIterationScore = latestResult.score;
		Score highBoundOffset = ASPIRATION_WINDOW_HALF_SIZE;
		Score lowBoundOffset = ASPIRATION_WINDOW_HALF_SIZE;

		// we are not adding an exception for the first iteration here because
		// it's very fast anyways

		bool highBoundOpen = false;
		bool lowBoundOpen = false;

		while (!m_context.Stopping())
		{
			latestResult.score = Search(
				m_context,
				latestResult.pv,
				m_context.startBoard,
				lowBoundOpen ? SCORE_MIN : (lastIterationScore - lowBoundOffset),
				highBoundOpen ? SCORE_MAX : (lastIterationScore + highBoundOffset),
				depth,
				0);

			if (latestResult.score >= (lastIterationScore + highBoundOffset) && !highBoundOpen)
			{
				// if we failed high, relax the upper bound
				highBoundOffset *= ASPIRATION_WINDOW_WIDEN_MULTIPLIER;

				if (highBoundOffset > ASPIRATION_WINDOW_HALF_SIZE_THRESHOLD)
				{
					highBoundOpen = true;
				}
			}
			else if (latestResult.score <= (lastIterationScore - lowBoundOffset) && !lowBoundOpen)
			{
				// if we failed low, relax the lower bound
				lowBoundOffset *= ASPIRATION_WINDOW_WIDEN_MULTIPLIER;

				if (lowBoundOffset > ASPIRATION_WINDOW_HALF_SIZE_THRESHOLD)
				{
					lowBoundOpen = true;
				}
			}
			else
			{
				// we are in window, so we are done!
				break;
			}
		}

		if (!m_context.Stopping())
		{
			m_rootResult = latestResult;

			ThinkingOutput thinkingOutput;
			thinkingOutput.nodeCount = m_context.nodeCount;
			thinkingOutput.ply = depth;

			// build the text pv
			Board b = m_context.startBoard;
			for (auto const &mv : latestResult.pv)
			{
				thinkingOutput.pv += b.MoveToAlg(mv) + ' ';
			}

			thinkingOutput.score = latestResult.score;
			thinkingOutput.time = CurrentTime() - startTime;

			m_context.thinkingOutputFunc(thinkingOutput);
		}

		m_context.onePlyDone = true;

		double elapsedTime = CurrentTime() - startTime;
		double totalAllocatedTime = endTime - startTime;
		double estimatedNextIterationTime = elapsedTime * ESTIMATED_MIN_BRANCHING_FACTOR;

		if (estimatedNextIterationTime > (totalAllocatedTime - elapsedTime) && m_context.searchType != SearchType_infinite)
		{
			break;
		}
	}

	Abort(); // interrupt timing thread

	if (m_searchTimerThread.joinable())
	{
		// this thread is only started if we are in time limited search
		m_searchTimerThread.join();
	}

	if (m_context.searchType == SearchType_makeMove)
	{
		std::string bestMove = m_context.startBoard.MoveToAlg(m_rootResult.pv[0]);
		m_context.finalMoveFunc(bestMove);
	}

	m_done = true;
}

void AsyncSearch::SearchTimer_(double time)
{
	// we have to do all this math because of GCC (libstdc++) bug #58038: http://gcc.gnu.org/bugzilla/show_bug.cgi?id=58038
	double endTime = CurrentTime() + time;

	std::unique_lock<std::mutex> lock(m_abortingMutex);

	while (!m_context.stopRequest && CurrentTime() < endTime)
	{
		// all this work for an interruptible sleep...
		// don't use wait_until here, because of the libstdc++ bug
		// we have to make sure wait time is positive
		double timeTillEnd = endTime - CurrentTime();

		if (timeTillEnd > 0.0)
		{
			m_cvAborting.wait_for(lock, std::chrono::microseconds(static_cast<uint64_t>(timeTillEnd * 1000000)));
		}
	}

	m_context.stopRequest = true;
}

Score Search(RootSearchContext &context, std::vector<Move> &pv, Board &board, Score alpha, Score beta, Depth depth, int32_t ply, bool nullMoveAllowed)
{
	// switch to QSearch if we are at depth 0
	if (depth <= 0)
	{
		return QSearch(context, pv, board, alpha, beta, ply);
	}

	++context.nodeCount;

	if (context.Stopping())
	{
		// if global stop request is set, we just return any value since it won't be used anyways
		return 0;
	}

	pv.clear();

	// we have to check for draws before probing the transposition table, because the ttable
	// can potentially hide repetitions

	// first we check for hard draws
	if (board.HasInsufficientMaterial())
	{
		return DRAW_SCORE;
	}

	// now we check for soft draws (only if ply > 0)
	if (ply > 0 && (board.Is2Fold(NUM_MOVES_TO_LOOK_FOR_DRAW) || board.Is50Moves()))
	{
		return DRAW_SCORE;
	}

	Depth originalDepth = depth;

	bool isPV = (beta - alpha) != 1;

	bool isRoot = ply == 0;

	// we cannot probe at root because then we would have no move to return
	if (!isRoot)
	{
		GTB::ProbeResult gtbResult = GTB::Probe(board);

		if (gtbResult)
		{
			return *gtbResult;
		}
	}

	TTEntry *tEntry = ENABLE_TT ? context.transpositionTable->Probe(board.GetHash()) : 0;

	// if we are at a PV node and don't have a best move (either because we don't have an entry,
	// or the entry doesn't have a best move)
	// internal iterative deepening
	if (ENABLE_IID && ENABLE_TT)
	{
		if (isPV && (!tEntry || tEntry->bestMove == 0) && depth > 4)
		{
			std::vector<Move> iidPv;
			Search(context, iidPv, board, alpha, beta, depth - 2, ply);

			tEntry = context.transpositionTable->Probe(board.GetHash());
		}
	}

	if (tEntry)
	{
		// try to get a cutoff from ttable, unless we are in PV (it can shorten PV)
		if (tEntry->depth >= depth && !isPV)
		{
			if (tEntry->entryType == EXACT)
			{
				// if we have an exact score, we can always return it
				return tEntry->score;
			}
			else if (tEntry->entryType == UPPERBOUND)
			{
				// if we have an upper bound, we can only return if this score fails low (no best move)
				if (tEntry->score <= alpha)
				{
					return tEntry->score;
				}
			}
			else if (tEntry->entryType == LOWERBOUND)
			{
				// if we have an upper bound, we can only return if this score fails high
				if (tEntry->score >= beta)
				{
					return tEntry->score;
				}
			}
		}
	}

	Score staticEval = context.evaluator->EvaluateForSTM(board, alpha, beta);

	// try null move
	if (ENABLE_NULL_MOVE_HEURISTICS && staticEval >= beta && !isPV)
	{
		Depth reduction = NULL_MOVE_REDUCTION;

		if (depth > 1 && !board.InCheck() && !board.IsZugzwangProbable() && nullMoveAllowed)
		{
			board.MakeNullMove();

			std::vector<Move> pvNN;
			Score nmScore = -Search(context, pvNN, board, -beta, -beta + 1, depth - reduction, ply + 1, false);

			board.UndoMove();

			if (nmScore >= beta)
			{
				if (ENABLE_TT)
				{
					context.transpositionTable->Store(board.GetHash(), 0, nmScore, originalDepth, LOWERBOUND);
				}

				return beta;
			}
		}
	}

	MoveEvaluatorIface::MoveInfoList miList;

	MoveEvaluatorIface::SearchInfo si;

	if (tEntry)
	{
		si.hashMove = tEntry->bestMove;
	}

	if (ENABLE_KILLERS)
	{
		si.killer = context.killer;
	}

	si.isQS = false;
	si.ply = ply;

	context.moveEvaluator->GenerateAndEvaluateMoves(board, si, miList);

	if (miList.GetSize() == 0)
	{
		// the game has ended
		if (board.InCheck())
		{
			// if we are in check and have no move to make, we are mated
			return MATE_OPPONENT_SIDE;
		}
		else
		{
			// otherwise this is a stalemate
			return 0;
		}
	}

	int numMovesSearched = -1;

	std::vector<Move> subPv;

	bool alphaRaised = false;

	for (auto &mi : miList)
	{
		Move mv = mi.move;

		board.ApplyMove(mv);

		++numMovesSearched;

		Score score = 0;

		// only search the first move with full window, since everything else is expected to fail low
		// if this is a null window search anyways, don't bother
		if (ENABLE_PVS && numMovesSearched != 0 && ((beta - alpha) != 1) && depth > 1)
		{
			score = -Search(context, subPv, board, -alpha - 1, -alpha, depth - 1, ply + 1);

			if (score > alpha && score < beta)
			{
				// if the move didn't actually fail low, this is now the PV, and we have to search with
				// full window
				score = -Search(context, subPv, board, -beta, -alpha, depth - 1, ply + 1);
			}
		}
		else
		{
			score = -Search(context, subPv, board, -beta, -alpha, depth - 1, ply + 1);
		}

		board.UndoMove();

		if (context.Stopping())
		{
			return 0;
		}

		AdjustIfMateScore(score);

		if (score > alpha)
		{
			alpha = score;
			alphaRaised = true;
			pv.clear();
			pv.push_back(ClearScore(mv));
			pv.insert(pv.end(), subPv.begin(), subPv.end());
		}

		if (score >= beta)
		{
			if (ENABLE_TT)
			{
				context.transpositionTable->Store(board.GetHash(), ClearScore(mv), score, originalDepth, LOWERBOUND);
			}

			// we don't want to store captures because those are searched before killers anyways
			if (!board.IsViolent(mv))
			{
				context.killer->Notify(ply, ClearScore(mv));
			}

			return score;
		}
	}

	if (!context.Stopping())
	{
		if (alphaRaised)
		{
			// if we have a bestMove, that means we have a PV node
			if (ENABLE_TT)
			{
				context.transpositionTable->Store(board.GetHash(), pv[0], alpha, originalDepth, EXACT);
			}

			if (!board.IsViolent(pv[0]))
			{
				context.killer->Notify(ply, pv[0]);
			}
		}
		else
		{
			// otherwise we failed low
			if (ENABLE_TT)
			{
				context.transpositionTable->Store(board.GetHash(), 0, alpha, originalDepth, UPPERBOUND);
			}
		}
	}

	return alpha;
}

Score QSearch(RootSearchContext &context, std::vector<Move> &pv, Board &board, Score alpha, Score beta, int32_t ply)
{
	++context.nodeCount;

	if (context.Stopping())
	{
		// if global stop request is set, we just return any value since it won't be used anyways
		return 0;
	}

	// if we are in check, switch back to normal search (we have to do this before stand-pat)
	if (board.InCheck() && ply < 20)
	{
		return Search(context, pv, board, alpha, beta, 1, ply, false);
	}

	pv.clear();

	// in QSearch we are only worried about insufficient material
	if (board.HasInsufficientMaterial())
	{
		return DRAW_SCORE;
	}

	GTB::ProbeResult gtbResult = GTB::Probe(board);

	if (gtbResult)
	{
		return *gtbResult;
	}

	// we first see if we can stand-pat
	Score staticEval = context.evaluator->EvaluateForSTM(board, alpha, beta);

	if (staticEval >= beta)
	{
		return staticEval;
	}

	bool isPV = (beta - alpha) != 1;

	TTEntry *tEntry = ENABLE_TT ? context.transpositionTable->Probe(board.GetHash()) : 0;

	if (tEntry)
	{
		// try to get a cutoff from ttable, unless we are in PV (it can shorten PV)
		// since we are in Q-search, we don't have to check depth
		if (!isPV)
		{
			if (tEntry->entryType == EXACT)
			{
				// if we have an exact score, we can always return it
				return tEntry->score;
			}
			else if (tEntry->entryType == UPPERBOUND)
			{
				// if we have an upper bound, we can only return if this score fails low (no best move)
				if (tEntry->score <= alpha)
				{
					return tEntry->score;
				}
			}
			else if (tEntry->entryType == LOWERBOUND)
			{
				// if we have an upper bound, we can only return if this score fails high
				if (tEntry->score >= beta)
				{
					return tEntry->score;
				}
			}
		}
	}

	// if we weren't able to get a cutoff, we may still be able to raise alpha to save some work
	if (staticEval > alpha)
	{
		alpha = staticEval;
	}

	MoveEvaluatorIface::MoveInfoList miList;

	MoveEvaluatorIface::SearchInfo si;

	if (tEntry)
	{
		si.hashMove = tEntry->bestMove;
	}

	if (ENABLE_KILLERS)
	{
		si.killer = context.killer;
	}

	si.isQS = true;
	si.ply = ply;

	context.moveEvaluator->GenerateAndEvaluateMoves(board, si, miList);

	std::vector<Move> subPv;

	for (auto &mi : miList)
	{
		if (mi.nodeAllocation == 0.0f)
		{
			continue;
		}

		Move mv = mi.move;

#ifdef DEBUG
		Score seeScoreCalculated = SEE::StaticExchangeEvaluation(board, mv);
		// extract the SEE score
		Score seeScore = GetScoreBiased(mv);

		assert(seeScore == seeScoreCalculated);
#endif
		board.ApplyMove(mv);

		Score score = 0;

		score = -QSearch(context, subPv, board, -beta, -alpha, ply + 1);

		board.UndoMove();

		if (context.Stopping())
		{
			return 0;
		}

		if (score > alpha)
		{
			alpha = score;
			pv.clear();
			pv.push_back(ClearScore(mv));
			pv.insert(pv.end(), subPv.begin(), subPv.end());
		}

		if (score >= beta)
		{
			return score;
		}
	}

	return alpha;
}

SearchResult SyncSearchDepthLimited(const Board &b, Depth depth, EvaluatorIface *evaluator, MoveEvaluatorIface *moveEvaluator, Killer *killer, TTable *ttable)
{
	SearchResult ret;
	RootSearchContext context;

	context.startBoard = b;

	std::unique_ptr<Killer> killer_u;
	std::unique_ptr<TTable> ttable_u;

	if (killer == nullptr)
	{
		killer_u.reset(new Killer);
		context.killer = killer_u.get();
	}
	else
	{
		context.killer = killer;
	}

	if (ttable == nullptr)
	{
		ttable_u.reset(new TTable(4*KB));
		context.transpositionTable = ttable_u.get();
	}
	else
	{
		context.transpositionTable = ttable;
	}

	context.evaluator = evaluator;
	context.moveEvaluator = moveEvaluator;

	context.searchType = SearchType_infinite;
	context.maxDepth = depth;

	context.stopRequest = false;
	context.onePlyDone = false;

	ret.score = Search(context, ret.pv, context.startBoard, SCORE_MIN, SCORE_MAX, depth, 0);

	return ret;
}

}
