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

#include "search.h"

#include <utility>
#include <memory>
#include <atomic>
#include <chrono>

#include <cstdint>

#include "history.h"
#include "types.h"
#include "util.h"
#include "eval/eval.h"
#include "see.h"
#include "gtb.h"
#include "countermove.h"

namespace
{
	// estimated minimum branching factor for time allocation
	// if more than 1/x of the allocated time has been used at the end of an iteration,
	// a new iteration won't be started
	const static double ESTIMATED_MIN_BRANCHING_FACTOR = 1.0;

	// how much to increase node budget by in each iteration of ID
	const static float NodeBudgetMultiplier = 4.0f;

	// with node count based search we can potentially go very deep, so we
	// have to call it a day at some point to avoid stack overflow
	const static Search::Depth MaxRecursionDepth = 64;
}

namespace Search
{

// this is mostly just to prevent overflow
// this number multiplied by maximum node budget multiplier must be less than 2^63
// it needs to be very high because node budget != node count (it also includes node counts for prunned nodes)
static const NodeBudget ID_MAX_NODE_BUDGET = 200000000000000000LL;

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

	if (m_context.nodeBudget == 0 || m_context.nodeBudget > ID_MAX_NODE_BUDGET)
	{
		m_context.nodeBudget = ID_MAX_NODE_BUDGET;
	}

	m_context.onePlyDone = false;

	latestResult.score = 0;

	int32_t iteration = 0;

	MoveList ml;
	m_context.startBoard.GenerateAllLegalMoves<Board::ALL>(ml);

	for (NodeBudget nodeBudget = 1;
			(nodeBudget <= m_context.nodeBudget) &&
			((CurrentTime() < endTime) || (m_context.searchType == SearchType_infinite) || !m_context.onePlyDone) &&
			(!m_context.Stopping());
		 nodeBudget *= NodeBudgetMultiplier)
	{
		++iteration;

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
				nodeBudget,
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
				// we are in window, so we are done (for this iteration)!
				break;
			}
		}

		if (!m_context.Stopping())
		{
			m_rootResult = latestResult;

			ThinkingOutput thinkingOutput;
			thinkingOutput.nodeCount = m_context.nodeCount;
			thinkingOutput.ply = iteration;

			// build the text pv
			Board b = m_context.startBoard;
			for (auto const &mv : latestResult.pv)
			{
				thinkingOutput.pv += b.MoveToAlg(mv) + ' ';
			}

			thinkingOutput.score = latestResult.score;
			thinkingOutput.time = CurrentTime() - startTime;

			m_context.thinkingOutputFunc(thinkingOutput);

			std::cout << "# d: " << iteration <<
						 " node budget: " << nodeBudget <<
						 " NPS: " << (static_cast<float>(m_context.nodeCount) / thinkingOutput.time) << std::endl;
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

Score Search(RootSearchContext &context, std::vector<Move> &pv, Board &board, Score alpha, Score beta, NodeBudget nodeBudget, int32_t ply, bool nullMoveAllowed)
{
	bool isPV = (beta - alpha) != 1;

	pv.clear();

	// switch to QSearch if we are out of nodes
	// using < 1 guarantees that a root search with nodeBudget 1 will always do a full ply
	if (nodeBudget < 1 || ply > MaxRecursionDepth)
	{
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
				else if (tEntry->entryType == UPPERBOUND && tEntry->score <= alpha)
				{
					return tEntry->score;
				}
				else if (tEntry->entryType == LOWERBOUND &&tEntry->score >= beta)
				{
					return tEntry->score;
				}
			}
		}

		Score ret = QSearch(context, pv, board, alpha, beta, ply, 0);

		// we want to store first ply q-search results
		if (!context.Stopping())
		{
			if (ret >= beta)
			{
				context.transpositionTable->Store(board.GetHash(), pv.size() > 0 ? pv[0] : 0, ret, 0, LOWERBOUND);
			}
			else if (ret <= alpha)
			{
				context.transpositionTable->Store(board.GetHash(), 0, ret, 0, UPPERBOUND);
			}
			else
			{
				context.transpositionTable->Store(board.GetHash(), pv.size() > 0 ? pv[0] : 0, ret, 0, EXACT);
			}
		}

		return ret;
	}

	++context.nodeCount;

	if (context.Stopping())
	{
		// if global stop request is set, we just return any value since it won't be used anyways
		return 0;
	}

	// we have to check for draws before probing the transposition table, because the ttable
	// can potentially hide repetitions

	// first we check for hard draws (if ply > 0, we use relaxed rules, which we can't do at ply = 0, because
	// it would result in empty pv)
	if (board.HasInsufficientMaterial(ply > 0))
	{
		return DRAW_SCORE;
	}

	// now we check for soft draws (only if ply > 0)
	if (ply > 0 && (board.Is2Fold(NUM_MOVES_TO_LOOK_FOR_DRAW) || board.Is50Moves()))
	{
		return DRAW_SCORE;
	}

	NodeBudget originalNodeBudget = nodeBudget;

	--nodeBudget; // for this node

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
		if (isPV && (!tEntry || tEntry->bestMove == 0) && nodeBudget > MinNodeBudgetForIID)
		{
			std::vector<Move> iidPv;
			Search(context, iidPv, board, alpha, beta, nodeBudget * IIDNodeBudgetMultiplier, ply);

			tEntry = context.transpositionTable->Probe(board.GetHash());
		}
	}

	if (tEntry)
	{
		// try to get a cutoff from ttable, unless we are in PV (it can shorten PV)
		if (tEntry->nodeBudget >= nodeBudget && !isPV)
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
		if (nodeBudget >= MinNodeBudgetForNullMove && !board.InCheck() && !board.IsZugzwangProbable() && nullMoveAllowed)
		{
			board.MakeNullMove();

			std::vector<Move> pvNN;

			NodeBudget nmNodeBudget = nodeBudget * NullMoveNodeBudgetMultiplier;

			Score nmScore = -Search(context, pvNN, board, -beta, -beta + 1, nmNodeBudget, ply + 1, false);

			board.UndoMove();

			if (nmScore >= beta)
			{
				if (ENABLE_TT)
				{
					context.transpositionTable->Store(board.GetHash(), 0, nmScore, originalNodeBudget, LOWERBOUND);
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

	if (ENABLE_COUNTERMOVES)
	{
		si.counter = context.counter;
	}

	if (ENABLE_HISTORY)
	{
		si.history = context.history;
	}

	si.isQS = false;
	si.ply = ply;
	si.tt = context.transpositionTable;
	si.totalNodeBudget = nodeBudget;

	si.lowerBound = alpha;
	si.upperBound = beta;

	auto searchFunc = [&context](Board &pos, Score lowerBound, Score upperBound, int64_t nodeBudget, int32_t ply) -> Score
	{
		std::vector<Move> pv;
		return Search(context, pv, pos, lowerBound, upperBound, nodeBudget, ply, true);
	};

	si.searchFunc = searchFunc;

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

	// we keep track of bestScore separately to fail soft on alpha
	Score bestScore = std::numeric_limits<Score>::min();

	for (auto &mi : miList)
	{		
		Move mv = mi.move;

		++numMovesSearched;

		// this means move evaluator wants to prune this move
		if (mi.nodeAllocation == 0.0f)
		{
			continue;
		}

		board.ApplyMove(mv);

		NodeBudget childNodeBudget = nodeBudget * mi.nodeAllocation;

		Score score = 0;

		// don't go into QS directly if in check (meaning the move we are searching is a checking move)
		if (board.InCheck())
		{
			childNodeBudget = std::max<NodeBudget>(childNodeBudget, 1);
		}

		// only search the first move with full window, since everything else is expected to fail low
		// if this is a null window search anyways, don't bother
		if (ENABLE_PVS && numMovesSearched != 0 && ((beta - alpha) != 1) && nodeBudget > MinNodeBudgetForPVS)
		{
			score = -Search(context, subPv, board, -alpha - 1, -alpha, childNodeBudget, ply + 1);

			if (score > alpha && score < beta)
			{
				// if the move didn't actually fail low, this is now the PV, and we have to search with
				// full window
				score = -Search(context, subPv, board, -beta, -alpha, childNodeBudget, ply + 1);
			}
		}
		else
		{
			score = -Search(context, subPv, board, -beta, -alpha, childNodeBudget, ply + 1);
		}

		board.UndoMove();

		if (context.Stopping())
		{
			return 0;
		}

		AdjustIfMateScore(score);

		if (score > bestScore)
		{
			bestScore = score;
			pv.clear();
			pv.push_back(mv);
			pv.insert(pv.end(), subPv.begin(), subPv.end());
		}

		if (score > alpha)
		{
			alpha = score;
		}

		if (score >= beta)
		{
			if (ENABLE_TT)
			{
				context.transpositionTable->Store(board.GetHash(), mv, score, originalNodeBudget, LOWERBOUND);
			}

			context.moveEvaluator->NotifyBestMove(board, si, miList, mv, numMovesSearched + 1);

			// we don't want to store captures because those are searched before killers anyways
			if (!board.IsViolent(mv))
			{
				if (ENABLE_KILLERS)
				{
					context.killer->Notify(ply, mv);
				}

				if (ENABLE_COUNTERMOVES)
				{
					context.counter->Notify(board, mv);
				}

				if (ENABLE_HISTORY)
				{
					context.history->NotifyCutoff(mv, originalNodeBudget);
				}
			}

			return score;
		}
		else
		{
			if (ENABLE_HISTORY)
			{
				assert(context.history);
				context.history->NotifyNoCutoff(mv, originalNodeBudget);
			}
		}
	}

	if (!context.Stopping())
	{
		if (bestScore > alpha)
		{
			if (ENABLE_TT)
			{
				context.transpositionTable->Store(board.GetHash(), pv[0], bestScore, originalNodeBudget, EXACT);
			}

			context.moveEvaluator->NotifyBestMove(board, si, miList, pv[0], miList.GetSize());
		}
		else
		{
			// otherwise we failed low (and may have prunned all nodes)
			if (ENABLE_TT)
			{
				context.transpositionTable->Store(board.GetHash(), pv.size() > 0 ? pv[0] : 0, bestScore, originalNodeBudget, UPPERBOUND);
			}
		}
	}

	return bestScore;
}

Score QSearch(RootSearchContext &context, std::vector<Move> &pv, Board &board, Score alpha, Score beta, int32_t ply, int32_t qsPly)
{
	++context.nodeCount;

	pv.clear();

	if (context.Stopping())
	{
		// if global stop request is set, we just return any value since it won't be used anyways
		return 0;
	}

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

	// if we are in check, and this is not the first ply in QS, switch back to normal search
	// we have to make sure it's not the first ply because otherwise if a leaf is in check, we can
	// get an explosion
	if (board.InCheck() && qsPly > 0)
	{
		return Search(context, pv, board, alpha, beta, 1, ply, true);
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

	if (ENABLE_COUNTERMOVES)
	{
		si.counter = context.counter;
	}

	if (ENABLE_HISTORY)
	{
		si.history = context.history;
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

		score = -QSearch(context, subPv, board, -beta, -alpha, ply + 1, qsPly + 1);

		board.UndoMove();

		if (context.Stopping())
		{
			return 0;
		}

		if (score > alpha)
		{
			alpha = score;
			pv.clear();
			pv.push_back(mv);
			pv.insert(pv.end(), subPv.begin(), subPv.end());
		}

		if (score >= beta)
		{
			return score;
		}
	}

	return alpha;
}

SearchResult SyncSearchNodeLimited(const Board &b, NodeBudget nodeBudget, EvaluatorIface *evaluator, MoveEvaluatorIface *moveEvaluator, Killer *killer, TTable *ttable, CounterMove *counter, History *history)
{
	SearchResult ret;
	RootSearchContext context;

	context.startBoard = b;

	std::unique_ptr<Killer> killer_u;
	std::unique_ptr<TTable> ttable_u;
	std::unique_ptr<CounterMove> counter_u;
	std::unique_ptr<History> history_u;

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

	if (counter == nullptr)
	{
		counter_u.reset(new CounterMove);
		context.counter = counter_u.get();
	}
	else
	{
		context.counter = counter;
	}

	if (history == nullptr)
	{
		history_u.reset(new History);
		context.history = history_u.get();
	}
	else
	{
		context.history = history;
	}

	context.evaluator = evaluator;
	context.moveEvaluator = moveEvaluator;

	context.searchType = SearchType_infinite;
	context.nodeBudget = nodeBudget;

	context.stopRequest = false;
	context.onePlyDone = false;

	ret.score = Search(context, ret.pv, context.startBoard, SCORE_MIN, SCORE_MAX, nodeBudget, 0);

	return ret;
}

bool trace = false;

}
