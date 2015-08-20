#include "ann_move_evaluator.h"
#include "ann_move_evaluator.h"

#include "random_device.h"
#include "search.h"
#include "static_move_evaluator.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

namespace
{

void TargetsToYNN(const std::vector<float> &trainingTargets, NNMatrixRM &yNN)
{
	yNN.resize(trainingTargets.size(), 1);

	for (int64_t i = 0; i < static_cast<int64_t>(trainingTargets.size()); ++i)
	{
		yNN(i, 0) = trainingTargets[i];
	}
}

}

ANNMoveEvaluator::ANNMoveEvaluator(ANNEvaluator &annEval)
	: m_annEval(annEval)
{
	std::vector<FeaturesConv::FeatureDescription> fds;

	FeaturesConv::GetMovesFeatureDescriptions(fds);

	m_ann = LearnAnn::BuildMoveEvalNet(fds.size(), 1);
}

void ANNMoveEvaluator::Train(const std::vector<std::string> &positions, const std::vector<std::string> &bestMoves)
{
	NNMatrixRM trainingSet;
	std::vector<float> trainingTarget;

	// training set size is approx 35 * positionsPerBatch
	size_t positionsPerBatch = std::min<size_t>(positions.size(), 16);

	const static size_t NumIterations = 10000;
	const static size_t IterationsPerPrint = 100;

	auto rng = gRd.MakeMT();
	auto positionDist = std::uniform_int_distribution<size_t>(0, positions.size() - 1);
	auto positionDrawFunc = std::bind(positionDist, rng);

	for (size_t iter = 0; iter < NumIterations; ++iter)
	{
		if ((iter % IterationsPerPrint) == 0)
		{
			std::cout << iter << "/" << NumIterations << std::endl;
		}

		trainingSet.resize(0, 0);
		trainingTarget.clear();

		for (size_t positionNum = 0; positionNum < positionsPerBatch; ++positionNum)
		{
			size_t idx = positionDrawFunc();

			Board pos(positions[idx]);
			Move bestMove = pos.ParseMove(bestMoves[idx]);

			MoveList ml;
			pos.GenerateAllLegalMoves<Board::ALL>(ml);

			NNMatrixRM trainingSetBatch;
			std::vector<float> trainingTargetBatch;

			FeaturesConv::ConvertMovesInfo convInfo;

			SearchInfo si;

			auto searchFunc = [this](Board &pos, Score /*lowerBound*/, Score /*upperBound*/, int64_t nodeBudget, int32_t /*ply*/) -> Score
			{
				Search::SearchResult result = Search::SyncSearchNodeLimited(pos, nodeBudget, &m_annEval, &gStaticMoveEvaluator);

				return result.score;
			};

			si.totalNodeBudget = 10000;

			si.searchFunc = searchFunc;

			GenerateMoveConvInfo_(pos, ml, convInfo);

			FeaturesConv::ConvertMovesToNN(pos, convInfo, ml, trainingSetBatch);

			for (size_t moveNum = 0; moveNum < ml.GetSize(); ++moveNum)
			{
				if (bestMove == ml[moveNum])
				{
					trainingTargetBatch.push_back(1.0f);
				}
				else
				{
					trainingTargetBatch.push_back(0.0f);
				}
			}

			assert(static_cast<size_t>(trainingSetBatch.rows()) == trainingTargetBatch.size());

			#pragma omp critical(trainingSetInsert)
			{
				int64_t origNumExamples = trainingSet.rows();

				NNMatrixRM orig = trainingSet;

				trainingSet.resize(trainingSet.rows() + trainingSetBatch.rows(), trainingSetBatch.cols());

				if (origNumExamples != 0)
				{
					// we have to copy the original over again because resize invalidates everything
					trainingSet.block(0, 0, origNumExamples, trainingSet.cols()) = orig;
				}

				trainingSet.block(origNumExamples, 0, trainingSetBatch.rows(), trainingSet.cols()) = trainingSetBatch;

				trainingTarget.insert(trainingTarget.end(), trainingTargetBatch.begin(), trainingTargetBatch.end());
			}
		}

		NNMatrixRM yNN;
		TargetsToYNN(trainingTarget, yNN);

		assert(trainingSet.rows() == yNN.rows());

		m_ann.TrainGDM(trainingSet, yNN, 1.0f, 0.0f);
	}
}

void ANNMoveEvaluator::Test(const std::vector<std::string> &positions, const std::vector<std::string> &bestMoves)
{
	// where in the list is the best move found
	int64_t orderPosCount[100] = { 0 };

	float averageConfidence = 0.0f;

	for (size_t posNum = 0; posNum < positions.size(); ++posNum)
	{
		SearchInfo si;
		MoveInfoList list;
		MoveList ml;

		Board board(positions[posNum]);
		board.GenerateAllLegalMoves<Board::ALL>(ml);

		for (size_t i = 0; i < ml.GetSize(); ++i)
		{
			MoveInfo mi;
			mi.move = ml[i];
			list.PushBack(mi);
		}

		si.totalNodeBudget = 10000;

		auto searchFunc = [this](Board &pos, Score /*lowerBound*/, Score /*upperBound*/, int64_t nodeBudget, int32_t /*ply*/) -> Score
		{
			Search::SearchResult result = Search::SyncSearchNodeLimited(pos, nodeBudget, &m_annEval, &gStaticMoveEvaluator);

			return result.score;
		};

		si.searchFunc = searchFunc;

		EvaluateMoves(board, si, list, ml);

		NormalizeMoveInfoList(list);

		Move bestMove = board.ParseMove(bestMoves[posNum]);

		assert(list.GetSize() == ml.GetSize());

		for (size_t i = 0; i < list.GetSize(); ++i)
		{
			if (bestMove == list[i].move)
			{
				if (i < 100)
				{
					++orderPosCount[i];
				}

				averageConfidence += list[i].nodeAllocation;
			}
		}
	}

	averageConfidence /= positions.size();

	std::cout << "Ordering position: " << std::endl;

	int64_t cCount = 0;

	for (size_t i = 0; i < 20; ++i)
	{
		cCount += orderPosCount[i];

		std::cout << i << ": " << (static_cast<float>(orderPosCount[i]) / positions.size() * 100.0f) << "%" <<
			" (" << (static_cast<float>(cCount) / positions.size() * 100.0f) << ")" << std::endl;
	}

	std::cout << "Average Confidence: " << averageConfidence << std::endl;
}

void ANNMoveEvaluator::EvaluateMoves(Board &board, SearchInfo &si, MoveInfoList &list, MoveList &ml)
{
	if (si.isQS || si.totalNodeBudget < MinimumNodeBudget)
	{
		// delegate to the static evaluator if it's QS, or if we are close to leaf
		// since we don't want to spend more time deciding what to search than actually searching them
		gStaticMoveEvaluator.EvaluateMoves(board, si, list, ml);
		return;
	}

	if (ml.GetSize() <= 1)
	{
		return;
	}

	FeaturesConv::ConvertMovesInfo convInfo;

	GenerateMoveConvInfo_(board, ml, convInfo);

	NNMatrixRM xNN;

	FeaturesConv::ConvertMovesToNN(board, convInfo, ml, xNN);

	NNMatrixRM results = m_ann.ForwardPropagateFast(xNN);

	float maxAllocation = 0.0f;

	float maxAllocationNp = 0.0f; // max allocation for moves with SEE <= 0

	for (int64_t i = 0; i < results.rows(); ++i)
	{
		list[i].nodeAllocation = results(i);

		list[i].seeScore = SEE::StaticExchangeEvaluation(board, list[i].move);

		if (results(i) > maxAllocation)
		{
			maxAllocation = results(i);
		}

		if (list[i].seeScore <= 0 && results(i) > maxAllocationNp)
		{
			maxAllocationNp = results(i);
		}
	}

	KillerMoveList killerMoves;

	if (si.killer)
	{
		si.killer->GetKillers(killerMoves, si.ply);
	}

	// now we go through the list again, and apply knowledge from search context (hash move and killers)
	for (size_t i = 0; i < list.GetSize(); ++i)
	{
		if (list[i].move == si.hashMove)
		{
			// bring the hash move to front
			list[i].nodeAllocation = 1.5f * maxAllocation;
		}
		else if (killerMoves.Exists(list[i].move) && !board.IsViolent(list[i].move))
		{
			// bring them to probably somewhere between winning captures and all other moves
			for (size_t slot = 0; slot < killerMoves.GetSize(); ++slot)
			{
				if (killerMoves[slot] == list[i].move)
				{
					// for killer moves, score is based on which slot we are in (lower = better)
					list[i].nodeAllocation = maxAllocationNp * (1.01f - 0.0001f * slot);

					break;
				}
			}
		}
	}

	NormalizeMoveInfoList(list);

	std::stable_sort(list.begin(), list.end(), [&si, &board](const MoveInfo &a, const MoveInfo &b)
		{
			if (a.move == si.hashMove)
			{
				return true;
			}
			else if (b.move == si.hashMove)
			{
				return false;
			}

			if (a.seeScore >= 0 && board.IsViolent(a.move) && (b.seeScore < 0 || !board.IsViolent(b.move)))
			{
				return true;
			}
			else if (b.seeScore >= 0 && board.IsViolent(b.move) && (a.seeScore < 0 || !board.IsViolent(a.move)))
			{
				return false;
			}

			return a.nodeAllocation > b.nodeAllocation;
		}
	);
}

void ANNMoveEvaluator::PrintDiag(Board &b)
{
	SearchInfo si;
	si.isQS = false;

	si.totalNodeBudget = 100000;

	auto searchFunc = [this](Board &pos, Score /*lowerBound*/, Score /*upperBound*/, int64_t nodeBudget, int32_t /*ply*/) -> Score
	{
		Search::SearchResult result = Search::SyncSearchNodeLimited(pos, nodeBudget, &m_annEval, &gStaticMoveEvaluator);

		return result.score;
	};

	si.searchFunc = searchFunc;

	MoveInfoList list;

	GenerateAndEvaluateMoves(b, si, list);

	for (auto &mi : list)
	{
		std::cout << b.MoveToAlg(mi.move) << ": " << mi.nodeAllocation << std::endl;
	}
}

void ANNMoveEvaluator::Serialize(std::ostream &os)
{
	SerializeNet(m_ann, os);
}

void ANNMoveEvaluator::Deserialize(std::istream &is)
{
	DeserializeNet(m_ann, is);
}

void ANNMoveEvaluator::GenerateMoveConvInfo_(Board &board, MoveList &ml, FeaturesConv::ConvertMovesInfo &convInfo)
{
	convInfo.see.resize(ml.GetSize());

	for (size_t i = 0; i < ml.GetSize(); ++i)
	{
		convInfo.see[i] = SEE::StaticExchangeEvaluation(board, ml[i]);
	}
}
