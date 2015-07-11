#include "learn.h"

#include <stdexcept>
#include <vector>
#include <sstream>
#include <algorithm>
#include <random>

#include <cmath>

#include <omp.h>

#include "matrix_ops.h"
#include "board.h"
#include "ann/features_conv.h"
#include "ann/learn_ann.h"
#include "omp_scoped_thread_limiter.h"
#include "eval/eval.h"
#include "search.h"
#include "ttable.h"
#include "killer.h"
#include "random_device.h"
#include "ann/ann_evaluator.h"

namespace
{

using namespace Learn;

std::string getFilename(int64_t iter)
{
	std::stringstream filenameSs;

	filenameSs << "trainingResults/eval" << iter << ".net";

	return filenameSs.str();
}

bool fileExists(const std::string &filename)
{
	std::ifstream is(filename);

	return is.good();
}

// plays a game from start position, and store the leaves of each position in leaves, and the scores in whiteScores
void playGame(std::vector<Board> &leaves, std::vector<float> &whiteScores, EvaluatorIface *evaluatorPtr, Killer &killer, TTable &ttable, std::mt19937 &mt)
{
	Board board;

	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	auto drawFunc = [&mt, &dist]() -> float { return dist(mt); };

	// this is a fast clear that just increments age of the table
	ttable.ClearTable();

	while (board.GetGameStatus() == Board::ONGOING && leaves.size() < MaxHalfmovesPerGame)
	{
		// first generate all legal moves from this position
		std::vector<Move> legalMoves;

		MoveList ml;
		board.GenerateAllLegalMovesSlow<Board::ALL>(ml);

		for (size_t i = 0; i < ml.GetSize(); ++i)
		{
			legalMoves.push_back(ml[i]);
		}

		// search results after making each move (scores are from opponent's POV!)
		std::vector<Search::SearchResult> results(legalMoves.size());

		// do a fixed depth search on each possible move, and record score
		for (size_t moveNum = 0; moveNum < legalMoves.size(); ++moveNum)
		{
			board.ApplyMove(legalMoves[moveNum]);

			results[moveNum] = Search::SyncSearchDepthLimited(board, SearchDepth, evaluatorPtr, &killer, &ttable);

			board.UndoMove();
		}

		// now we pick a move based on how much worse they are than the best move
		float bestScoreSTM = std::numeric_limits<float>::lowest();

		for (size_t moveNum = 0; moveNum < legalMoves.size(); ++moveNum)
		{
			bestScoreSTM = std::max(bestScoreSTM, results[moveNum].score * -1.0f); // -1 because it's from opponent's POV
		}

		std::vector<float> moveProbabilities(legalMoves.size());

		for (size_t moveNum = 0; moveNum < legalMoves.size(); ++moveNum)
		{
			float scoreSTM = results[moveNum].score * -1.0f;

			// with the new scaling, a pawn is about 2000
			moveProbabilities[moveNum] = pow(0.98f, (bestScoreSTM - scoreSTM) / 20.0f);
		}

		/*
		std::cout << board.GetFen() << std::endl;
		for (size_t i = 0; i < moveProbabilities.size(); ++i)
		{
			std::cout << board.MoveToAlg(legalMoves[i]) << ' ' << moveProbabilities[i] << std::endl;
		}
		std::cout << std::endl;
		*/

		// normalize the probabilities
		float sum = 0.0f;
		sum = std::accumulate(moveProbabilities.begin(), moveProbabilities.end(), 0.0f);

		std::for_each(moveProbabilities.begin(), moveProbabilities.end(), [sum](float &x) { x /= sum; });

		// draw a move to make
		float numDrawn = drawFunc();

		size_t moveToMake = 0;

		float cumProb = 0.0f;

		for (size_t moveNum = 0; moveNum < legalMoves.size(); ++moveNum)
		{
			cumProb += moveProbabilities[moveNum];

			if (cumProb >= numDrawn)
			{
				moveToMake = moveNum;
				break;
			}
		}

		// make the move, and store the leaf position and score
		board.ApplyMove(legalMoves[moveToMake]);

		Board leaf = board;
		leaf.ApplyVariation(results[moveToMake].pv);

		float whiteScore = results[moveToMake].score * (board.GetSideToMove() == WHITE ? 1.0f : -1.0f);

		whiteScore = evaluatorPtr->UnScale(whiteScore);

		leaves.push_back(leaf);
		whiteScores.push_back(whiteScore); // no need to flip here

		killer.MoveMade();
		ttable.AgeTable();
	}
}

}

namespace Learn
{

void TDLSelfPlay()
{
	std::cout << "Starting TDL training..." << std::endl;

	// get feature descriptions
	std::vector<FeaturesConv::FeatureDescription> featureDescriptions;

	FeaturesConv::ConvertBoardToNN(Board(), featureDescriptions);

	ANNEvaluator annEvaluator;

	int64_t iter = 0;

	// try to read existing dumps to continue where we left off
	for (; iter < NumIterations; ++iter)
	{
		std::string filename = getFilename(iter);

		if (!fileExists(filename))
		{
			break;
		}
	}

	--iter;

	std::string filename = getFilename(iter);
	std::ifstream dump(filename);
	annEvaluator.Deserialize(dump);

	if (iter != 0)
	{
		std::cout << "Continuing from iteration " << iter << std::endl;
	}

	for (; iter < NumIterations; ++iter)
	{
		std::cout << "======= Iteration: " << iter << " =======" << std::endl;

		std::cout << "Generating training set for this iteration..." << std::endl;

		std::vector<std::string> trainingPositions; // store as FEN to save memory
		std::vector<float> trainingTargets;

		size_t gamesPlayed = 0;
		double timeStart = CurrentTime();
		double timeLastPrint = CurrentTime();

		#pragma omp parallel
		{
			Killer killer;
			TTable ttable(256*KB);

			ttable.InvalidateAllEntries();

			auto thread_mt = gRd.MakeMT();

			// we have to make a copy of the evaluator because it's not re-entrant (due to caching)
			ANNEvaluator thread_evaluator = annEvaluator;

			EvaluatorIface *evaluator = (iter == 0) ? static_cast<EvaluatorIface *>(&Eval::gStaticEvaluator) : static_cast<EvaluatorIface *>(&thread_evaluator);

			// first generate the training set by playing a bunch of games
			#pragma omp for schedule(dynamic, 1)
			for (int64_t game = 0; game < ((iter == 0) ? GamesFirstIteration : GamesPerIteration); ++game)
			{
				std::vector<Board> leaves;
				std::vector<float> whiteScores;

				playGame(leaves, whiteScores, evaluator, killer, ttable, thread_mt);

				std::vector<float> differences(leaves.size() - 1);

				// compute the differences between each adjacent score
				for (size_t i = 0; i < differences.size(); ++i)
				{
					differences[i] = whiteScores[i + 1] - whiteScores[i];
				}

				// apply the TD equation to get errors for each position, and compute training target
				// we store them locally first to minimize time in critical section
				std::vector<std::string> trainingPositionsFromThisGame;
				std::vector<float> trainingTargetsFromThisGame;

				for (size_t i = 0; i < leaves.size() ; ++i)
				{
					// only add to training set if the leaf eval is the same as position eval (not mates, etc)
					auto staticEval = evaluator->UnScale(evaluator->EvaluateForWhite(leaves[i]));
					if (staticEval != whiteScores[i])
					{
						continue;
					}

					float totalError = 0.0f;
					float discountFactor = 1.0f;

					for (size_t j = i; j < differences.size(); ++j)
					{
						totalError += differences[j] * discountFactor;
						discountFactor *= Lambda;
					}

					trainingPositionsFromThisGame.push_back(leaves[i].GetFen());
					trainingTargetsFromThisGame.push_back(staticEval + totalError * LearningRate);
				}

				#pragma omp critical(trainingPositionsInsert)
				{
					trainingPositions.insert(trainingPositions.end(), trainingPositionsFromThisGame.begin(), trainingPositionsFromThisGame.end());
					trainingTargets.insert(trainingTargets.end(), trainingTargetsFromThisGame.begin(), trainingTargetsFromThisGame.end());
				}

				if (omp_get_thread_num() == 0)
				{
					float timeSinceLastPrint = CurrentTime() - timeLastPrint;
					if (timeSinceLastPrint > 15.0f)
					{
						double timeElapsed = CurrentTime() - timeStart;
						std::cout << "Games played: " << gamesPlayed << " Games/s: " << (gamesPlayed / timeElapsed) << std::endl;
						timeLastPrint = CurrentTime();
					}
				}

				#pragma omp atomic
				++gamesPlayed;
			}
		}

		std::cout << "Num training positions: " << trainingPositions.size() << std::endl;

		// now we can convert the boards to neural network representation
		NNMatrixRM trainingTargetsNN(static_cast<int64_t>(trainingPositions.size()), 1);

		std::cout << "Generating training set took " << (CurrentTime() - timeStart) << " seconds" << std::endl;

		std::cout << "Updating weights..." << std::endl;

		if (iter == 0)
		{
			annEvaluator.BuildANN(std::string("x_5M_nodiag.features"), featureDescriptions.size());

			annEvaluator.TrainLoop(trainingPositions, trainingTargetsNN, 30, featureDescriptions);
		}
		else
		{
			annEvaluator.Train(trainingPositions, trainingTargetsNN, featureDescriptions);
		}

		std::ofstream annOut(getFilename(iter));

		annEvaluator.Serialize(annOut);

		/*
		for (size_t i = 0; i < trainingPositions.size(); ++i)
		{
			std::cout << trainingPositions[i] << std::endl;
			std::cout << trainingTargets[i] << " " << annEvaluator.UnScale(annEvaluator.EvaluateForWhite(Board(trainingPositions[i]))) << std::endl;
		}
		*/

		std::cout << "Iteration took " << (CurrentTime() - timeStart) << " seconds" << std::endl;
	}
}

void TDL(const std::string &positionsFilename)
{
	std::cout << "Starting TDL training..." << std::endl;

	std::ifstream positionsFile(positionsFilename);

	if (!positionsFile)
	{
		throw std::runtime_error(std::string("Cannot open ") + positionsFilename + " for reading");
	}

	// these are the root positions for training (they don't change)
	std::vector<std::string> rootPositions;

	std::string fen;

	std::cout << "Reading FENs..." << std::endl;

	while (std::getline(positionsFile, fen))
	{
		rootPositions.push_back(fen);
	}

	std::cout << "Positions read: " << rootPositions.size() << std::endl;

	// these are the leaf positions used in training
	// they are initialized to root positions, but will change in second iteration
	std::vector<std::string> trainingPositions(PositionsPerBatch);

	NNMatrixRM trainingTargets(trainingPositions.size(), 1);

	ANNEvaluator annEvaluator;

	std::vector<FeaturesConv::FeatureDescription> featureDescriptions;

	FeaturesConv::ConvertBoardToNN(Board(), featureDescriptions);

	int64_t iter = 0;

	// try to read existing dumps to continue where we left off
	for (; iter < NumIterations; ++iter)
	{
		std::string filename = getFilename(iter);

		if (!fileExists(filename))
		{
			break;
		}
	}

	if (iter > 0)
	{
		--iter;

		std::string filename = getFilename(iter);
		std::ifstream dump(filename);
		annEvaluator.Deserialize(dump);
	}

	if (iter != 0)
	{
		std::cout << "Continuing from iteration " << iter << std::endl;
	}

	for (; iter < NumIterations; ++iter)
	{
		std::cout << "======= Iteration: " << iter << " =======" << std::endl;

		std::random_shuffle(rootPositions.begin(), rootPositions.end());

		// first we generate new labels
		// if this is the first iteration, we use static material labels
		if (iter == 0)
		{
			std::cout << "Labelling using static evaluation..." << std::endl;

			trainingPositions.resize(PositionsFirstBatch);
			trainingTargets.resize(trainingPositions.size(), 1);

			#pragma omp parallel for
			for (size_t i = 0; i < PositionsFirstBatch; ++i)
			{
				Board b(rootPositions[i]);
				Score val = Eval::gStaticEvaluator.EvaluateForWhite(b, SCORE_MIN, SCORE_MAX);
				trainingPositions[i] = rootPositions[i];
				trainingTargets(i, 0) = Eval::gStaticEvaluator.UnScale(val);
			}
		}
		else
		{
			std::cout << "Labelling using TDLeaf..." << std::endl;

			size_t positionsProcessed = 0;
			double timeStart = CurrentTime();
			double timeLastPrint = CurrentTime();

			trainingPositions.resize(PositionsPerBatch);
			trainingTargets.resize(trainingPositions.size(), 1);

			#pragma omp parallel
			{
				// each thread has her own ttable and killers, to save on page faults and allocations/deallocations
				Killer thread_killer;
				TTable thread_ttable(1*MB); // we want the ttable to fit in L3

				// we are being paranoid here - it's possible tha the memory we get happens to be where a TTable used to be,
				// in which case we will have many valid entries with wrong scores (since evaluator changed)
				thread_ttable.InvalidateAllEntries();

				// each thread makes a copy of the evaluator to reduce sharing
				ANNEvaluator thread_annEvaluator = annEvaluator;

				#pragma omp for schedule(dynamic, 8)
				for (size_t i = 0; i < PositionsPerBatch; ++i)
				{
					thread_ttable.ClearTable(); // this is a cheap clear that simply ages the table a bunch so all new positions have higher priority

					Board rootPos(rootPositions[i]);
					Search::SearchResult rootResult = Search::SyncSearchDepthLimited(rootPos, SearchDepth, &thread_annEvaluator, &thread_killer, &thread_ttable);

					Board leafPos = rootPos;
					leafPos.ApplyVariation(rootResult.pv);

					float leafScore = thread_annEvaluator.EvaluateForWhite(leafPos); // this should theoretically be the same as the search result, except for mates, etc

					float rootScoreWhite = rootResult.score * (rootPos.GetSideToMove() == WHITE ? 1.0f : -1.0f);

					trainingPositions[i] = leafPos.GetFen();

					float leafScoreUnscaled = thread_annEvaluator.UnScale(leafScore);

					if (rootResult.pv.size() > 0 && (leafScore == rootScoreWhite))
					{
						rootPos.ApplyMove(rootResult.pv[0]);
						thread_killer.MoveMade();
						thread_ttable.AgeTable();

						// now we compute the error by making a few moves
						float accumulatedError = 0.0f;
						float lastScore = leafScoreUnscaled;
						float discount = 1.0f;

						for (int64_t m = 0; m < HalfMovesToMake; ++m)
						{
							Search::SearchResult result = Search::SyncSearchDepthLimited(rootPos, SearchDepth, &thread_annEvaluator, &thread_killer, &thread_ttable);

							float scoreWhiteUnscaled = thread_annEvaluator.UnScale(result.score * (rootPos.GetSideToMove() == WHITE ? 1.0f : -1.0f));

							// compute error contribution
							accumulatedError += discount * (scoreWhiteUnscaled - lastScore);
							lastScore = scoreWhiteUnscaled;
							discount *= Lambda;

							if ((rootPos.GetGameStatus() != Board::ONGOING) || (result.pv.size() == 0))
							{
								break;
							}

							rootPos.ApplyMove(result.pv[0]);
							thread_killer.MoveMade();
							thread_ttable.AgeTable();
						}

						accumulatedError = std::max(accumulatedError, -MaxError);
						accumulatedError = std::min(accumulatedError, MaxError);

						trainingTargets(i, 0) = leafScoreUnscaled + LearningRate * accumulatedError;
					}
					else
					{
						// if PV is empty or leaf score is not the same as search score, this is an end position, and we don't need to train it
						trainingTargets(i, 0) = thread_annEvaluator.UnScale(leafScore);
					}

					if (omp_get_thread_num() == 0)
					{
						float timeSinceLastPrint = CurrentTime() - timeLastPrint;
						if (timeSinceLastPrint > 15.0f)
						{
							double timeElapsed = CurrentTime() - timeStart;
							std::cout << "Processed: " << positionsProcessed << " Positions/s: " << (positionsProcessed / timeElapsed) << std::endl;
							timeLastPrint = CurrentTime();
						}
					}

					#pragma omp atomic
					++positionsProcessed;
				}
			}
		}

		std::cout << "Updating weights..." << std::endl;

		if (iter == 0)
		{
			annEvaluator.BuildANN(std::string("x_5M_nodiag.features"), featureDescriptions.size());

			annEvaluator.TrainLoop(trainingPositions, trainingTargets, 1, featureDescriptions);
		}
		else
		{
			annEvaluator.Train(trainingPositions, trainingTargets, featureDescriptions);
		}

		std::ofstream annOut(getFilename(iter));

		annEvaluator.Serialize(annOut);
	}
}

}
