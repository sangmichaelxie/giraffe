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

#include "learn.h"

#include <stdexcept>
#include <vector>
#include <sstream>
#include <algorithm>
#include <random>
#include <functional>

#include <cmath>

#include <omp.h>

#include "matrix_ops.h"
#include "board.h"
#include "ann/features_conv.h"
#include "ann/learn_ann.h"
#include "omp_scoped_thread_limiter.h"
#include "eval/eval.h"
#include "history.h"
#include "search.h"
#include "ttable.h"
#include "killer.h"
#include "countermove.h"
#include "random_device.h"
#include "ann/ann_evaluator.h"
#include "move_evaluator.h"
#include "static_move_evaluator.h"
#include "util.h"
#include "stats.h"

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

}

namespace Learn
{

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
		assert(fen != "");
	}

	std::cout << "Positions read: " << rootPositions.size() << std::endl;

	// these are the leaf positions used in training
	// they are initialized to root positions, but will change in second iteration
	std::vector<std::string> trainingPositions(PositionsPerBatch);

	NNMatrixRM trainingTargets(trainingPositions.size(), 1);

	ANNEvaluator annEvaluator;

	std::vector<FeaturesConv::FeatureDescription> featureDescriptions;

	Board dummyBoard;
	FeaturesConv::ConvertBoardToNN(dummyBoard, featureDescriptions);

	int64_t iter = 0;

	// try to read existing dumps to continue where we left off
	for (; iter < NumIterations; iter += EvaluatorSerializeInterval)
	{
		std::string filename = getFilename(iter);

		if (!fileExists(filename))
		{
			break;
		}
	}

	if (iter > 0)
	{
		iter -= EvaluatorSerializeInterval;

		std::string filename = getFilename(iter);
		std::ifstream dump(filename);
		annEvaluator.Deserialize(dump);
	}

	if (iter != 0)
	{
		std::cout << "Continuing from iteration " << iter << std::endl;
	}

	double timeStart = CurrentTime();

	Stat errorStat;

	for (; iter < NumIterations; ++iter)
	{
		double iterationStart = CurrentTime();

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
			size_t positionsProcessed = 0;

			trainingPositions.resize(PositionsPerBatch);
			trainingTargets.resize(trainingPositions.size(), 1);

			#pragma omp parallel
			{
				// each thread has her own ttable, killers, and counter, to save on page faults and allocations/deallocations
				Killer thread_killer;
				TTable thread_ttable(1*MB); // we want the ttable to fit in L3
				CounterMove thread_counter;
				History thread_history;

				// we are being paranoid here - it's possible tha the memory we get happens to be where a TTable used to be,
				// in which case we will have many valid entries with wrong scores (since evaluator changed)
				thread_ttable.InvalidateAllEntries();

				// each thread makes a copy of the evaluator to reduce sharing
				ANNEvaluator thread_annEvaluator = annEvaluator;

				auto rng = gRd.MakeMT();
				auto positionDist = std::uniform_int_distribution<size_t>(0, rootPositions.size() - 1);
				auto positionDrawFunc = std::bind(positionDist, rng);

				#pragma omp for schedule(dynamic, 1)
				for (size_t i = 0; i < PositionsPerBatch; ++i)
				{
					thread_ttable.ClearTable(); // this is a cheap clear that simply ages the table a bunch so all new positions have higher priority

					Board rootPos(rootPositions[positionDrawFunc()]);

					if (rootPos.GetGameStatus() != Board::ONGOING)
					{
						continue;
					}

					//if (realDrawFunc() < 0.3f)
					{
						// make 1 random move
						// it's very important that we make an odd number of moves, so that if the move is something stupid, the
						// opponent can take advantage of it (and we will learn that this position is bad) before we have a chance to
						// fix it
						MoveList ml;
						rootPos.GenerateAllLegalMoves<Board::ALL>(ml);

						auto movePickerDist = std::uniform_int_distribution<size_t>(0, ml.GetSize() - 1);

						rootPos.ApplyMove(ml[movePickerDist(rng)]);

						if (rootPos.GetGameStatus() != Board::ONGOING)
						{
							continue;
						}
					}

					Search::SearchResult rootResult = Search::SyncSearchNodeLimited(rootPos, SearchNodeBudget, &thread_annEvaluator, &gStaticMoveEvaluator, &thread_killer, &thread_ttable, &thread_counter, &thread_history);

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
						thread_history.NotifyMoveMade();

						// now we compute the error by making a few moves
						float accumulatedError = 0.0f;
						float lastScore = leafScoreUnscaled;
						float tdDiscount = 1.0f;
						float absoluteDiscount = AbsLambda;

						for (int64_t m = 0; m < HalfMovesToMake; ++m)
						{
							Search::SearchResult result = Search::SyncSearchNodeLimited(rootPos, SearchNodeBudget, &thread_annEvaluator, &gStaticMoveEvaluator, &thread_killer, &thread_ttable, &thread_counter, &thread_history);

							float scoreWhiteUnscaled = thread_annEvaluator.UnScale(result.score * (rootPos.GetSideToMove() == WHITE ? 1.0f : -1.0f)) * absoluteDiscount;

							absoluteDiscount *= AbsLambda;

							// compute error contribution (only if same side)
							if (m % 2 == 1)
							{
								accumulatedError += tdDiscount * (scoreWhiteUnscaled - lastScore);
								lastScore = scoreWhiteUnscaled;
								tdDiscount *= TDLambda;
							}

							if ((rootPos.GetGameStatus() != Board::ONGOING) || (result.pv.size() == 0))
							{
								break;
							}

							rootPos.ApplyMove(result.pv[0]);
							thread_killer.MoveMade();
							thread_ttable.AgeTable();
							thread_history.NotifyMoveMade();
						}

						float absError = fabs(accumulatedError);

						#pragma omp critical(statUpdate)
						{
							errorStat.AddNumber(absError);
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

					#pragma omp atomic
					++positionsProcessed;
				}
			}
		}

		if (iter == 0)
		{
			annEvaluator.BuildANN(featureDescriptions.size());

			annEvaluator.TrainLoop(trainingPositions, trainingTargets, 1, featureDescriptions);
		}
		else
		{
			annEvaluator.Train(trainingPositions, trainingTargets, featureDescriptions, LearningRateSGD);
		}

		if ((iter % EvaluatorSerializeInterval) == 0)
		{
			auto mt = gRd.MakeMT();
			std::shuffle(rootPositions.begin(), rootPositions.end(), mt);

			std::cout << "Serializing..." << std::endl;

			std::ofstream annOut(getFilename(iter));

			annEvaluator.Serialize(annOut);
		}

		if ((iter % IterationPrintInterval) == 0)
		{
			std::cout << "Iteration " << iter << ". ";
			std::cout << "Time: " << (CurrentTime() - timeStart) << " seconds. ";
			std::cout << "Last Iteration took: " << (CurrentTime() - iterationStart) << " seconds. ";

			std::cout << "TD Error: " << errorStat.GetAvg() << ". ";
			errorStat.Reset();

			std::cout << std::endl;
		}
	}
}

}
