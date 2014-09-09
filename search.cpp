#include "search.h"

#include <utility>
#include <memory>
#include <atomic>
#include <chrono>

#include <cstdint>

#include "util.h"
#include "eval/eval.h"
#include "see.h"

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

	for (Depth depth = 1;
			(depth <= m_context.maxDepth) &&
			((CurrentTime() < endTime) || (m_context.searchType == SearchType_infinite) || !m_context.onePlyDone) &&
			(!m_context.Stopping());
		 ++depth)
	{
		latestResult.score = Search_(m_context, latestResult.bestMove, m_context.startBoard, SCORE_MIN, SCORE_MAX, depth, 0);

		if (!m_context.stopRequest || !m_context.onePlyDone)
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

		if (estimatedNextIterationTime > (totalAllocatedTime - elapsedTime))
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

	TTEntry *tEntry = context.transpositionTable->Probe(board.GetHash());

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

	bool legalMoveFound = false;

	// try null move
	if (depth > 3 && !board.InCheck() && !board.IsZugzwangProbable() && nullMoveAllowed)
	{
		board.MakeNullMove();

		Score nmScore = -Search_(context, board, -beta, -beta + 1, depth - 3 - 1, ply, false);

		board.UndoMove();

		if (nmScore >= beta)
		{
			// don't bother storing to TT here because the Search_ call would have stored it already
			return beta;
		}
	}

	MoveList moves;

	board.GenerateAllMoves<Board::ALL>(moves);

	bestMove = 0;

	// assign scores to all the moves
	for (size_t i = 0; i < moves.GetSize(); ++i)
	{
		// lower 16 bits are used for SEE value, biased by 0x8000
		uint32_t score = 0;

		bool seeEligible = board.IsSeeEligible(moves[i]);

		Score seeScore = 0;
		uint16_t biasedSeeScore = 0;

		if (seeEligible)
		{
			seeScore = StaticExchangeEvaluation(board, moves[i]);
			biasedSeeScore = seeScore + 0x8000;
		}

		uint32_t scoreType = 0;
		// upper [23:16] is move type
		// 255 = hash move
		// 254 = queen promotions
		// 253 = winning and equal captures
		// 252 = killer moves
		// 251 = all others (history heuristics?)
		// 250 = losing captures

		if (tEntry && moves[i] == tEntry->bestMove)
		{
			scoreType = 255;
		}
		else if (GetPromoType(moves[i]) == WQ)
		{
			scoreType = 254;
		}
		else if (seeEligible && seeScore >= 0)
		{
			scoreType = 253;
		}
		else if (!seeEligible)
		{
			scoreType = 251;
		}
		else
		{
			scoreType = 250;
		}

		score = biasedSeeScore + (scoreType << 16);

		SetScore(moves[i], score);
	}

	std::sort(moves.Begin(), moves.End(), [](const Move &a, const Move &b) { return a > b; });

	for (size_t i = 0; i < moves.GetSize(); ++i)
	{
		if (board.ApplyMove(moves[i]))
		{
			legalMoveFound = true;

			Score score = -Search_(context, board, -beta, -alpha, depth - 1, ply + 1);

			board.UndoMove();

			if (context.Stopping())
			{
				return 0;
			}

			AdjustIfMateScore(score);

			if (score > alpha)
			{
				alpha = score;
				bestMove = moves[i];
			}

			if (score >= beta)
			{
				context.transpositionTable->Store(board.GetHash(), ClearScore(moves[i]), score, depth, LOWERBOUND);
				return score;
			}
		}
	}

	if (legalMoveFound)
	{
		if (!context.Stopping())
		{
			if (bestMove)
			{
				// if we have a bestMove, that means we have a PV node
				context.transpositionTable->Store(board.GetHash(), ClearScore(bestMove), alpha, depth, EXACT);
			}
			else
			{
				// otherwise we failed low
				context.transpositionTable->Store(board.GetHash(), 0, alpha, depth, UPPERBOUND);
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

	// if we are in check, search all nodes using regular Search_
	if (board.InCheck())
	{
		return Search_(context, board, alpha, beta, 1, ply, true);
	}

	// if we are not in check, we first see if we can stand-pat
	Score staticEval = Eval::Evaluate(board, alpha, beta);

	if (staticEval >= beta)
	{
		return staticEval;
	}

	// if we cannot get a cutoff, we may still be able to raise alpha to save some work
	if (staticEval > alpha)
	{
		alpha = staticEval;
	}

	// now we start searching
	MoveList moves;

	board.GenerateAllMoves<Board::VIOLENT>(moves);

	// assign scores to all the moves for sorting
	for (size_t i = 0; i < moves.GetSize(); ++i)
	{
		// lower 16 bits are used for SEE value, biased by 0x8000
		uint32_t score = 0;

		bool seeEligible = board.IsSeeEligible(moves[i]);

		Score seeScore = 0;
		uint16_t biasedSeeScore = 0;

		if (seeEligible)
		{
			seeScore = StaticExchangeEvaluation(board, moves[i]);
			biasedSeeScore = seeScore + 0x8000;
		}

		uint32_t scoreType = 0;
		// upper [23:16] is move type
		// 254 = queen promotions
		// 253 = winning and equal captures
		// 250 = losing captures

		// since we are generating silent moves only, we should only get queen promotions
		if (GetPromoType(moves[i]) != 0)
		{
			scoreType = 254;
		}
		else if (seeEligible && seeScore >= 0)
		{
			scoreType = 253;
		}
		else
		{
			scoreType = 250;
		}

		score = biasedSeeScore + (scoreType << 16);

		SetScore(moves[i], score);
	}

	std::sort(moves.Begin(), moves.End(), [](const Move &a, const Move &b) { return a > b; });

	for (size_t i = 0; i < moves.GetSize(); ++i)
	{
		if (board.IsSeeEligible(moves[i]))
		{
			// extract the SEE score
			Score seeScore = 0;
			seeScore = GetScore(moves[i]);
			seeScore -= 0x8000; // unbias the score

			// only search the capture if it can potentially improve alpha
			if ((staticEval + seeScore) < alpha)
			{
				// we don't have to search any more moves, because moves were sorted by SEE
				break;
			}
		}

		if (board.ApplyMove(moves[i]))
		{
			Score score = -QSearch_(context, board, -beta, -alpha, ply + 1);

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
	}

	return alpha;
}

}
