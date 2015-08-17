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

void FeaturesToXNN(const std::vector<std::vector<float> >&features, NNMatrixRM &xNN)
{
	int64_t rows = features.size();
	int64_t cols = features[0].size();

	xNN.resize(rows, cols);

	for (int64_t row = 0; row < rows; ++row)
	{
		assert(static_cast<int64_t>(features[row].size()) == cols);

		for (int64_t col = 0; col < cols; ++col)
		{
			xNN(row, col) = features[row][col];
		}
	}
}

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
	std::vector<std::vector<float>> ret;
	Board b;

	MoveList ml;
	b.GenerateAllLegalMoves<Board::ALL>(ml);

	FeaturesConv::ConvertMovesInfo convInfo;

	FeaturesConv::ConvertMovesToNN(b, convInfo, ml, ret);

	m_ann = LearnAnn::BuildMoveEvalNet(ret[0].size(), 1);
}

void ANNMoveEvaluator::Train(const std::vector<std::string> &positions, const std::vector<std::string> &bestMoves)
{
	std::vector<std::vector<float>> trainingSet;
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

		trainingSet.clear();
		trainingTarget.clear();

		for (size_t positionNum = 0; positionNum < positionsPerBatch; ++positionNum)
		{
			size_t idx = positionDrawFunc();

			Board pos(positions[idx]);
			Move bestMove = pos.ParseMove(bestMoves[idx]);

			MoveList ml;
			pos.GenerateAllLegalMoves<Board::ALL>(ml);

			std::vector<std::vector<float>> trainingSetBatch;
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

			GenerateMoveConvInfo_(pos, ml, convInfo, m_annEval, si);

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

			assert(trainingSetBatch.size() == trainingTargetBatch.size());

			#pragma omp critical(trainingSetInsert)
			{
				trainingSet.insert(trainingSet.end(), trainingSetBatch.begin(), trainingSetBatch.end());
				trainingTarget.insert(trainingTarget.end(), trainingTargetBatch.begin(), trainingTargetBatch.end());
			}
		}

		NNMatrixRM xNN;
		FeaturesToXNN(trainingSet, xNN);

		NNMatrixRM yNN;
		TargetsToYNN(trainingTarget, yNN);

		m_ann.TrainGDM(xNN, yNN, 1.0f, 0.0f);
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

		std::sort(list.begin(), list.end(), [](const MoveInfo &a, const MoveInfo &b) { return a.nodeAllocation > b.nodeAllocation; });

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
	// call static evaluator now
	// in case of QS or low node budget, we will return right away
	// otherwise, we still use it for sorting (we will just overwrite the allocations)
	gStaticMoveEvaluator.EvaluateMoves(board, si, list, ml);

	if (si.isQS || si.totalNodeBudget < MinimumNodeBudget)
	{
		return;
	}

	if (ml.GetSize() == 0)
	{
		return;
	}

	// only do crazy stuff if we have a PV node
	if ((si.upperBound - si.lowerBound) <= 1)
	{
		return;
	}

	//Score expectedScore = si.searchFunc(board, si.lowerBound, si.upperBound, si.totalNodeBudget / 10000, si.ply);

	// since static evaluator only sorts the MoveInfoList, now we have to copy it back to ml
	assert(list.GetSize() == ml.GetSize());

	for (size_t i = 0; i < list.GetSize(); ++i)
	{
		ml[i] = list[i].move;
	}

	std::vector<std::vector<float>> features;

	FeaturesConv::ConvertMovesInfo convInfo;

	GenerateMoveConvInfo_(board, ml, convInfo, m_annEval, si);

	FeaturesConv::ConvertMovesToNN(board, convInfo, ml, features);

	// copy expected scores over to MoveInfoList for use later
	for (size_t i = 0; i < list.GetSize(); ++i)
	{
		list[i].expectedScore = convInfo.evalBefore + convInfo.evalDeltas[i];
	}

	float scoreParent = convInfo.evalBefore;

	NNMatrixRM xNN;
	FeaturesToXNN(features, xNN);

	NNMatrixRM results = m_ann.ForwardPropagateFast(xNN);

	float maxAllocation = 0.0f;

	for (int64_t i = 0; i < results.size(); ++i)
	{
		list[i].nodeAllocation *= std::max(std::min(results(i), 2.0f), 0.5f);

		if (results(i) > maxAllocation)
		{
			maxAllocation = results(i);
		}
	}

	NormalizeMoveInfoList(list);

	// apply minimum
	for (size_t i = 0; i < list.GetSize(); ++i)
	{
		list[i].nodeAllocation = std::max(0.01f, list[i].nodeAllocation);
	}

	/*

	float upperBoundUnscaled = m_annEval.UnScale(si.upperBound);
	float lowerBoundUnscaled = m_annEval.UnScale(si.lowerBound);

	KillerMoveList killerMoves;

	if (si.killer)
	{
		si.killer->GetKillers(killerMoves, si.ply);
	}

	std::stable_sort(list.begin(), list.end(), [&si, &upperBoundUnscaled, &lowerBoundUnscaled, &killerMoves, &scoreParent](const MoveInfo &a, const MoveInfo &b)
	{
		// hash move first
		if (a.move == si.hashMove)
		{
			return true;
		}

		if (b.move == si.hashMove)
		{
			return false;
		}

		return a.nodeAllocation > b.nodeAllocation;
	});
	*/

	// at this point, we have very good move ordering, and can now scale the allocations based on where they are,
	// and how the node allocations trend
	// we will not be normalizing afterwards
	// this is similar to LMR

	/*
	std::vector<float> scalings(list.GetSize());

	scalings[0] = 1.0f;

	for (size_t i = 1; i < list.GetSize(); ++i)
	{
		scalings[i] = scalings[i - 1] * 0.85f * std::min(std::max(list[i].nodeAllocation / list[i - 1].nodeAllocation, 0.8f), 1.0f);
	}

	for (size_t i = 0; i < list.GetSize(); ++i)
	{
		list[i].nodeAllocation *= scalings[i];
	}
	*/
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

void ANNMoveEvaluator::GenerateMoveConvInfo_(Board &board, MoveList &ml, FeaturesConv::ConvertMovesInfo &convInfo, ANNEvaluator &evaluator, SearchInfo &si)
{
	auto evalFunc = [&evaluator, &si](Board &board, Score lowerBound, Score upperBound, int64_t nodeBudget, int32_t ply) -> float
	{
		return evaluator.UnScale(si.searchFunc(board, lowerBound, upperBound, nodeBudget, ply));
	};

	//convInfo.evalBefore = evaluator.UnScale(evaluator.EvaluateForSTMGEE(board));
	convInfo.evalBefore = evalFunc(board, si.lowerBound, si.upperBound, si.totalNodeBudget / 10000, si.ply);

	for (size_t i = 0; i < ml.GetSize(); ++i)
	{
		board.ApplyMove(ml[i]);
		//float newScore = -evaluator.UnScale(evaluator.EvaluateForSTMGEE(board));
		float newScore = -evalFunc(board, -si.upperBound, -si.lowerBound, si.totalNodeBudget / 100000, si.ply + 1);
		convInfo.evalDeltas.push_back(newScore - convInfo.evalBefore);
		board.UndoMove();
	}
}
