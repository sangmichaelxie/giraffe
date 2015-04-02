#include "search.h"

#include <utility>
#include <memory>
#include <atomic>
#include <chrono>

#include <cstdint>

#include "util.h"
#include "eval/eval.h"
#include "see.h"
#include "movepicker.h"

namespace
{
	// a score of MATE_MOVING_SIDE means the opponent (of the moving side) is mated on the board
	const static Score MATE_MOVING_SIDE = 30000;

	// a score of MATE_OPPONENT_SIDE means the moving side is mated on the board
	const static Score MATE_OPPONENT_SIDE = -30000;

	const static Score MATE_MOVING_SIDE_THRESHOLD = 20000;
	const static Score MATE_OPPONENT_SIDE_THRESHOLD = -20000;

	// estimated minimum branching factor for time allocation
	// if more than 1/x of the allocated time has been used at the end of an iteration,
	// a new iteration won't be started
	const static double ESTIMATED_MIN_BRANCHING_FACTOR = 5.0;

	// when these mating scores are propagated up, they are adjusted by distance to mate
	inline void AdjustIfMateScore(Score &score)
	{
		if (score > MATE_MOVING_SIDE_THRESHOLD)
		{
			--score;
		}
		else if (score < MATE_OPPONENT_SIDE_THRESHOLD)
		{
			++score;
		}
	}

	inline bool IsMateScore(Score score)
	{
		return score > MATE_MOVING_SIDE_THRESHOLD || score < MATE_OPPONENT_SIDE_THRESHOLD;
	}
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
			latestResult.score = Search_(
				m_context,
				latestResult.bestMove,
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
			thinkingOutput.pv = m_context.startBoard.MoveToAlg(latestResult.bestMove);
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
		std::string bestMove = m_context.startBoard.MoveToAlg(m_rootResult.bestMove);
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

Score AsyncSearch::Search_(RootSearchContext &context, Move &bestMove, Board &board, Score alpha, Score beta, Depth depth, int32_t ply, bool nullMoveAllowed)
{
	// switch to QSearch is we are at depth 0
	if (depth <= 0)
	{
		return QSearch_(context, board, alpha, beta, ply);
	}

	++context.nodeCount;

	if (context.Stopping())
	{
		// if global stop request is set, we just return any value since it won't be used anyways
		return 0;
	}

	// we have to check for draws before probing the transposition table, because the ttable
	// can potentially hide repetitions
	if (ply > 0 && (board.Is2Fold(NUM_MOVES_TO_LOOK_FOR_DRAW) || board.Is50Moves()))
	{
		return DRAW_SCORE;
	}

	Depth originalDepth = depth;

	bool isPV = (beta - alpha) != 1;

	bool isRoot = ply == 0;

	TTEntry *tEntry = context.transpositionTable->Probe(board.GetHash());

	// if we are at a PV node and don't have a best move (either because we don't have an entry,
	// or the entry doesn't have a best move)
	// internal iterative deepening
	if (ENABLE_IID)
	{
		if (isPV && (!tEntry || tEntry->bestMove == 0) && depth > 4)
		{
			Move iidBestMove;
			Search_(context, iidBestMove, board, alpha, beta, depth - 2, ply);

			tEntry = context.transpositionTable->Probe(board.GetHash());
		}
	}

	if (tEntry)
	{
		// try to get a cutoff from ttable
		if (tEntry->depth >= depth)
		{
			if (tEntry->entryType == EXACT)
			{
				// if we have an exact score, we can always return it
				bestMove = tEntry->bestMove;
				return tEntry->score;
			}
			else if (tEntry->entryType == UPPERBOUND)
			{
				// if we have an upper bound, we can only return if this score fails low (no best move)
				if (tEntry->score <= alpha)
				{
					bestMove = 0;
					return tEntry->score;
				}
			}
			else if (tEntry->entryType == LOWERBOUND)
			{
				// if we have an upper bound, we can only return if this score fails high
				if (tEntry->score >= beta)
				{
					bestMove = tEntry->bestMove;
					return tEntry->score;
				}
			}
		}
	}

	// if we have a hit with insufficient depth, but enough to indicate that null move will be fruitless, skip it
	bool avoidNullTT = false;
	if (tEntry && (tEntry->entryType == UPPERBOUND || tEntry->entryType == EXACT) && tEntry->score < beta)
	{
		avoidNullTT = true;
	}

	// try null move
	if (ENABLE_NULL_MOVE_HEURISTICS)
	{
		Depth reduction = NULL_MOVE_REDUCTION;

		if (ENABLE_ADAPTIVE_NULL_MOVE && depth >= ADAPTIVE_NULL_MOVE_THRESHOLD)
		{
			reduction += 1;
		}

		if (depth > 1 && !board.InCheck() && (!board.IsZugzwangProbable() || NM_REDUCE_INSTEAD_OF_PRUNE) && nullMoveAllowed && !avoidNullTT)
		{
			board.MakeNullMove();

			Score nmScore = -Search_(context, board, -beta, -beta + 1, depth - reduction, ply + 1, false);

			board.UndoMove();

			if (nmScore >= beta)
			{
				if (NM_REDUCE_INSTEAD_OF_PRUNE)
				{
					depth -= NMR_DR;

					if (depth <= 0)
					{
						return QSearch_(context, board, alpha, beta, ply);
					}
				}
				else
				{
					context.transpositionTable->Store(board.GetHash(), 0, nmScore, originalDepth, LOWERBOUND);
					return beta;
				}
			}
		}
	}

	bestMove = 0;

	Score staticEval = Eval::Evaluate(board, alpha, beta);

	bool legalMoveFound = false;

	bool inCheck = board.InCheck();

	bool futilityAllowed = ENABLE_FUTILITY_PRUNING && !inCheck && !IsMateScore(alpha) && !IsMateScore(beta) && (depth < FUTILITY_MAX_DEPTH);

	Killer dummyKiller;

	MovePicker::MovePickerStage moveStage;

	MovePicker movePicker(board, tEntry ? tEntry->bestMove : 0, ENABLE_KILLERS ? *(context.killer) : dummyKiller, false, ply);

	Move mv = 0;

	int numMovesSearched = -1;

	while ((mv = movePicker.GetNextMove(moveStage)))
	{
		Score seeScore = GetScoreBiased(mv);

		if (board.ApplyMove(mv))
		{
			bool isViolent = board.IsViolent(mv);

			legalMoveFound = true;

			++numMovesSearched;

			int32_t extend = 0;
			// check extension
			// we have the depth > 1 condition here so we don't get qsearch explosion (since we search in-check positions
			// from QS at depth = 1)
			if (board.InCheck() && originalDepth > 1)
			{
				extend = 1;
			}

			// see if we can do futility pruning
			// futility pruning is when we are near the leaf, and are so far below alpha, that we only want to search
			// moves that can potentially improve alpha
			bool fut = !isRoot && futilityAllowed && !isViolent && !extend &&
						((staticEval + seeScore + FUTILITY_MARGINS[depth]) <= alpha);
			if (fut)
			{
				board.UndoMove();
				continue;
			}

			if (ENABLE_BAD_MOVE_PRUNING)
			{
				// if a move is in unlikely stage and is not a capture (just leaves a piece hanging),
				// and we are in the last few plies, just throw them away
				// TODO: find a way to make sure we don't return a1a1 as best move
				if (!isRoot && !inCheck && !isViolent && !extend && depth <= BAD_MOVE_PRUNING_MAX_DEPTH && moveStage == MovePicker::UNLIKELY)
				{
					board.UndoMove();
					continue;
				}
			}

			int32_t reduce = 0;

			if (ENABLE_LATE_MOVE_REDUCTION &&
				depth >= LMR_MIN_DEPTH &&
				!inCheck &&
				!extend &&
				moveStage != MovePicker::LIKELY &&
				numMovesSearched >= LMR_NUM_MOVES_FULL_DEPTH &&
				!isPV &&
				!isViolent)
			{
				// these are the Senpai rules
				if (numMovesSearched < LMR_NUM_MOVES_REDUCE_1)
				{
					reduce += LATE_MOVE_REDUCTION;
				}
				else
				{
					reduce += depth / 3;
				}

				// these are moves that leave a piece hanging (not captures, since we don't reduce captures,
				// even if they are losing captures)
				if (ENABLE_BAD_MOVE_REDUCTION && moveStage == MovePicker::UNLIKELY)
				{
					reduce += BAD_MOVE_REDUCTION;
				}
			}

			Score score = 0;

			// only search the first move with full window, since everything else is expected to fail low
			// if this is a null window search anyways, don't bother
			if (ENABLE_PVS && numMovesSearched != 0 && ((beta - alpha) != 1) && depth > 1)
			{
				score = -Search_(context, board, -alpha - 1, -alpha, depth - 1 + extend - reduce, ply + 1);

				if (score > alpha && score < beta && !reduce)
				{
					// if the move didn't actually fail low, this is now the PV, and we have to search with
					// full window
					// if we are reducing, then don't bother re-searching here, because we will be re-searching
					// below anyways
					score = -Search_(context, board, -beta, -alpha, depth - 1 + extend - reduce, ply + 1);
				}
			}
			else
			{
				score = -Search_(context, board, -beta, -alpha, depth - 1 + extend - reduce, ply + 1);
			}

			// if we reduced and the move turned out to not fail low, we should re-search at original depth
			// (and full window)
			if (reduce && score > alpha)
			{
				score = -Search_(context, board, -beta, -alpha, depth - 1 + extend, ply + 1);
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
				bestMove = ClearScore(mv);
			}

			if (score >= beta)
			{
				context.transpositionTable->Store(board.GetHash(), ClearScore(mv), score, originalDepth, LOWERBOUND);

				// we don't want to store captures because those are searched before killers anyways
				if (!board.IsViolent(mv))
				{
					context.killer->Notify(ply, ClearScore(mv));
				}

				return score;
			}
		}
	}

	// we don't have to check whether we are doing futility pruning here, because we still make those moves
	// anyways to check for legality
	if (legalMoveFound)
	{
		if (!context.Stopping())
		{
			if (bestMove)
			{
				// if we have a bestMove, that means we have a PV node
				context.transpositionTable->Store(board.GetHash(), bestMove, alpha, originalDepth, EXACT);

				if (!board.IsViolent(bestMove))
				{
					context.killer->Notify(ply, bestMove);
				}
			}
			else
			{
				// otherwise we failed low
				context.transpositionTable->Store(board.GetHash(), 0, alpha, originalDepth, UPPERBOUND);
			}
		}

		return alpha;
	}
	else
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
}

Score AsyncSearch::Search_(RootSearchContext &context, Board &board, Score alpha, Score beta, Depth depth, int32_t ply, bool nullMoveAllowed)
{
	Move dummy;
	return Search_(context, dummy, board, alpha, beta, depth, ply, nullMoveAllowed);
}

Score AsyncSearch::QSearch_(RootSearchContext &context, Board &board, Score alpha, Score beta, int32_t ply)
{
	++context.nodeCount;

	if (context.Stopping())
	{
		// if global stop request is set, we just return any value since it won't be used anyways
		return 0;
	}

	// we first see if we can stand-pat
	Score staticEval = Eval::Evaluate(board, alpha, beta);

	if (staticEval >= beta)
	{
		return staticEval;
	}

	// can any move possibly increase alpha?
	// we take opponent's biggest piece, and if even capturing that piece for free with the highest positional gain possible
	// doesn't improve alpha, there is no point going further
	// we also have to consider the possibility of promotion (when we have a pawn on 7th rank)
	// this is called delta pruning
	Score highestPossibleScore =
		staticEval +
		Eval::MAT[board.GetOpponentLargestPieceType()] +
		(board.HasPawnOn7th() ? Eval::MAT[WQ] : 0) +
		Eval::MAX_POSITIONAL_SCORE;
	if (highestPossibleScore <= alpha)
	{
		return highestPossibleScore;
	}

	// if we weren't able to get a cutoff, we may still be able to raise alpha to save some work
	if (staticEval > alpha)
	{
		alpha = staticEval;
	}

	MovePicker::MovePickerStage moveStage;

	// now we start searching
	MovePicker movePicker(board, 0, *(context.killer), true, ply);

	Move mv = 0;

	size_t i = 0;

	while ((mv = movePicker.GetNextMove(moveStage)))
	{
#ifdef DEBUG
		Score seeScoreCalculated = StaticExchangeEvaluation(board, mv);
#endif

		// extract the SEE score
		Score seeScore = GetScoreBiased(mv);
#ifdef DEBUG

		assert(seeScore == seeScoreCalculated);
#endif

		// only search the capture if it can potentially improve alpha
		PieceType promoType = GetPromoType(mv);
		Score promoVal = (promoType != 0) ? Eval::MAT[promoType & ~COLOR_MASK] : 0;
		if ((staticEval + seeScore + promoVal + Eval::MAX_POSITIONAL_SCORE) <= alpha)
		{
			continue;
		}

		// even if this move can potentially improve alpha, if it's a losing capture
		// we still don't search it, because it's highly likely that another capture or
		// standing pat will be better
		if (seeScore < 0)
		{
			continue;
		}

		if (board.ApplyMove(mv))
		{
			Score score = 0;

			score = -QSearch_(context, board, -beta, -alpha, ply + 1);

			board.UndoMove();

			if (context.Stopping())
			{
				return 0;
			}

			if (score > alpha)
			{
				alpha = score;
			}

			if (score >= beta)
			{
				return score;
			}
		}

		++i;
	}

	return alpha;
}

}
