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

#include "learn_ann.h"

#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <omp.h>

#include <cmath>

#include <sys/types.h>
#include <sys/stat.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <fcntl.h>

#include "ann.h"
#include "types.h"
#include "random_device.h"
#include "features_conv.h"
#include "consts.h"

namespace
{
const int64_t KMeanNumIterations = 1;

const size_t MaxBatchSize = 256;

const size_t MaxMemory = 32ULL*1024*1024*1024; // limit dataset size if we have many features

const size_t MaxIterationsPerCheck = 500000 / MaxBatchSize;

const float ExclusionFactor = 0.99f; // when computing test performance, ignore 1% of outliers

typedef std::vector<int32_t> Group;

struct Rows
{
	Rows() {}
	Rows(int64_t begin, int64_t num) : begin(begin), num(num) {}

	int64_t begin;
	int64_t num;
};

template <typename Derived1>
void SplitDataset(
	const Eigen::MatrixBase<Derived1> &x,
	Rows &train,
	Rows &val,
	Rows &test)
{
	size_t numExamples = x.rows();

	const float testRatio = 0.2f;
	const size_t MaxTest = 5000;
	const float valRatio = 0.2f;
	const size_t MaxVal = 5000;

	size_t testSize = std::min<size_t>(MaxTest, numExamples * testRatio);
	size_t valSize = std::min<size_t>(MaxVal, numExamples * valRatio);
	size_t trainSize = numExamples - testSize - valSize;

	test = Rows(0, testSize);

	val = Rows(testSize, valSize);

	train = Rows(testSize + valSize, trainSize);
}

template <typename T, typename Derived1, typename Derived2>
void Train(
	T &nn,
	int64_t epochs,
	Eigen::MatrixBase<Derived1> &xTrain,
	Eigen::MatrixBase<Derived2> &yTrain,
	Eigen::MatrixBase<Derived1> &xVal,
	Eigen::MatrixBase<Derived2> &yVal,
	Eigen::MatrixBase<Derived1> &/*xTest*/,
	Eigen::MatrixBase<Derived2> &/*yTest*/)
{
	size_t iter = 0;

	float trainingErrorAccum = 0.0f;

	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	T bestNet = nn;
	FP bestValScore = std::numeric_limits<FP>::max(); // this is updated every time val score improves

	bool done = false;

	size_t NumBatches = xTrain.rows() / MaxBatchSize;

	if ((xTrain.rows() % MaxBatchSize) != 0)
	{
		++NumBatches;
	}

	int64_t epoch = 0;

	// we want to check at least once per epoch
	size_t iterationsPerCheck = std::min(MaxIterationsPerCheck, NumBatches);

	size_t examplesSeen = 0;

	while (!done && epoch < epochs)
	{
		size_t batchNum = iter % NumBatches;

		size_t begin = batchNum * MaxBatchSize;

		size_t batchSize = std::min(MaxBatchSize, xTrain.rows() - begin);

		examplesSeen += batchSize;

		epoch = examplesSeen / xTrain.rows();

		trainingErrorAccum += nn.TrainGDM(
			xTrain.block(begin, 0, batchSize, xTrain.cols()),
			yTrain.block(begin, 0, batchSize, yTrain.cols()),
			1.0f,
			0.000001f);

		if ((iter % iterationsPerCheck) == 0)
		{
			NNMatrix pred = nn.ForwardPropagateFast(xVal);

			NNMatrix errors = nn.ErrorFunc(pred, yVal);

			FP valScore = errors.sum() / xVal.rows();

			if (valScore < bestValScore)
			{
				bestValScore = valScore;
				bestNet = nn;
			}

			std::cout << "Iteration: " << iter << ", ";

			std::cout << "Epoch: " << epoch << ", ";

			std::cout << "Val: " << valScore << ", ";

			std::cout << "Train: " << (trainingErrorAccum / std::min(iter + 1, iterationsPerCheck)) << ", ";

			std::chrono::seconds t = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - startTime);

			std::cout << "Time: " << (static_cast<float>(t.count()) / 60.0f) << " minutes, ";
			std::cout << "Best Val: " << bestValScore << ", ";

			std::cout << "Sparsity: " << nn.GetSparsity() << std::endl;

			trainingErrorAccum = 0.0f;
		}

		++iter;
	}

	nn = bestNet;
}

template <typename T>
NNMatrix EvaluateNet(T &nn, NNMatrixRM &x)
{
	// how many examples to evaluate at a time (memory restriction)
	const static int64_t ExamplesPerBatch = 2048;

	NNMatrix ret(x.rows(), nn.OutputCols());

	for (int64_t i = 0; i < x.rows();)
	{
		int64_t examplesToEval = std::min<int64_t>(x.rows() - i, ExamplesPerBatch);

		NNMatrix pred = nn.ForwardPropagateFast(x.block(i, 0, examplesToEval, x.cols()));

		ret.block(i, 0, examplesToEval, ret.cols()) = pred;

		i += examplesToEval;
	}

	return ret;
}

std::vector<std::tuple<size_t, size_t>> GetCombinations(size_t m /* for combinations of 0-10, use m = 11 */)
{
	std::vector<std::tuple<size_t, size_t>> ret;

	for (size_t elem0 = 0; elem0 < m; ++elem0)
	{
		for (size_t elem1 = 0; elem1 < elem0; ++elem1)
		{
			ret.push_back(std::make_tuple(elem0, elem1));
		}
	}

	return ret;
}

struct LayerDescription
{
	size_t layerSize;
	std::vector<Eigen::Triplet<float> > connections;

	LayerDescription() : layerSize(0) {}
};

void AddSingleNodesGroup(
	LayerDescription &layerDescription,
	const Group &groupIn,
	Group &groupOut,
	float nodeCountMultiplier
	)
{
	size_t nodesInGroup = groupIn.size();
	size_t nodesForThisGroup = ceil(nodesInGroup * nodeCountMultiplier);

	groupOut.clear();

	for (size_t i = 0; i < nodesForThisGroup; ++i)
	{
		for (auto feature : groupIn)
		{
			layerDescription.connections.push_back(Eigen::Triplet<float>(feature, layerDescription.layerSize, 1.0f));
		}

		groupOut.push_back(layerDescription.layerSize);

		++layerDescription.layerSize;
	}
}

void DebugPrintGroups(const std::vector<Group> &groups)
{
	std::cout << "Groups:" << std::endl;
	size_t groupNum = 0;
	for (auto group : groups)
	{
		std::cout << groupNum << " (" << group.size() << "): ";

		for (auto feature : group)
		{
			std::cout << feature << ' ';
		}

		std::cout << std::endl;

		++groupNum;
	}
}

void AnalyzeFeatureDescriptions(const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions,
								Group &globalGroup, /* global group does not include group 0! */
								Group &squareGroup,
								Group &group0)
{
	// first we make global feature groups
	for (size_t featureNum = 0; featureNum < featureDescriptions.size(); ++featureNum)
	{
		auto &fd = featureDescriptions[featureNum];

		if (fd.featureType == FeaturesConv::FeatureDescription::FeatureType_global)
		{
			if (fd.group == 0)
			{
				group0.push_back(featureNum);
			}
			else
			{
				globalGroup.push_back(featureNum);
			}
		}
		else if (fd.featureType == FeaturesConv::FeatureDescription::FeatureType_pos)
		{
			squareGroup.push_back(featureNum);
		}
	}

	//assert(squareGroup.size() == (2*64));
	assert(group0.size() > 5 && group0.size() < 40);
}

} // namespace

namespace LearnAnn
{

EvalNet BuildEvalNet(int64_t inputDims, int64_t outputDims, bool smallNet)
{
	std::vector<size_t> layerSizes;
	std::vector<std::vector<Eigen::Triplet<float> > > connMatrices;

	Group globalGroup;
	Group squareGroup;
	Group group0;

	// get feature descriptions
	std::vector<FeaturesConv::FeatureDescription> featureDescriptions;
	Board dummyBoard;
	FeaturesConv::ConvertBoardToNN(dummyBoard, featureDescriptions);

	AnalyzeFeatureDescriptions(featureDescriptions,
									globalGroup,
									squareGroup,
									group0);

	if (!smallNet)
	{
		LayerDescription layer0;

		Group layer0Group0;
		Group layer0GlobalGroup;
		Group layer0SquareGroup;

		// first we add the mixed global group
		AddSingleNodesGroup(layer0, globalGroup, layer0GlobalGroup, 0.05f);

		// mixed square group
		AddSingleNodesGroup(layer0, squareGroup, layer0SquareGroup, 0.05f);

		// pass through group 0 (this contains game phase information)
		AddSingleNodesGroup(layer0, group0, layer0Group0, 1.0f);

		layerSizes.push_back(layer0.layerSize);
		connMatrices.push_back(layer0.connections);

		// in the second layer, we just fully connect everything
		layerSizes.push_back(BoardSignatureSize);
		connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

		// fully connected output layer
		connMatrices.push_back(std::vector<Eigen::Triplet<float> >());
	}
	else
	{
		LayerDescription layer0;

		Group layer0Group0;
		Group layer0GlobalGroup;
		Group layer0SquareGroup;

		// first we add the mixed global group
		AddSingleNodesGroup(layer0, globalGroup, layer0GlobalGroup, 0.1f);

		// mixed square group
		AddSingleNodesGroup(layer0, squareGroup, layer0SquareGroup, 0.1f);

		// pass through group 0 (this contains game phase information)
		AddSingleNodesGroup(layer0, group0, layer0Group0, 1.0f);

		layerSizes.push_back(layer0.layerSize);
		connMatrices.push_back(layer0.connections);

		// in the second layer, we just fully connect everything
		layerSizes.push_back(BoardSignatureSize);
		connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

		// fully connected output layer
		connMatrices.push_back(std::vector<Eigen::Triplet<float> >());
	}

	return EvalNet(inputDims, outputDims, layerSizes, connMatrices);
}

MoveEvalNet BuildMoveEvalNet(int64_t inputDims, int64_t outputDims)
{
	std::vector<size_t> layerSizes;
	std::vector<std::vector<Eigen::Triplet<float> > > connMatrices;

	Group globalGroup;
	Group squareGroup;
	Group group0;

	// get feature descriptions
	std::vector<FeaturesConv::FeatureDescription> featureDescriptions;
	GetMovesFeatureDescriptions(featureDescriptions);

	AnalyzeFeatureDescriptions(featureDescriptions,
									globalGroup,
									squareGroup,
									group0);

	LayerDescription layer0;

	Group layer0Group0;
	Group layer0GlobalGroup;
	Group layer0SquareGroup;

	// first we add the mixed global group
	AddSingleNodesGroup(layer0, globalGroup, layer0GlobalGroup, 0.1f);

	// mixed square group
	AddSingleNodesGroup(layer0, squareGroup, layer0SquareGroup, 0.1f);

	// pass through group 0 (this contains game phase and move-specific information)
	AddSingleNodesGroup(layer0, group0, layer0Group0, 0.5f);

	layerSizes.push_back(layer0.layerSize);
	connMatrices.push_back(layer0.connections);

	// in the second layer, we just fully connect everything
	layerSizes.push_back(BoardSignatureSize);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	// fully connected output layer
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	return MoveEvalNet(inputDims, outputDims, layerSizes, connMatrices);
}

template <typename Derived1, typename Derived2>
void TrainANN(
	const Eigen::MatrixBase<Derived1> &x,
	const Eigen::MatrixBase<Derived2> &y,
	EvalNet &nn,
	int64_t epochs)
{
	Rows trainRows, valRows, testRows;
	SplitDataset(x, trainRows, valRows, testRows);

	auto xTrain = x.block(trainRows.begin, 0, trainRows.num, x.cols());
	auto yTrain = y.block(trainRows.begin, 0, trainRows.num, y.cols());
	auto xVal = x.block(valRows.begin, 0, valRows.num, x.cols());
	auto yVal = y.block(valRows.begin, 0, valRows.num, y.cols());
	auto xTest = x.block(testRows.begin, 0, testRows.num, x.cols());
	auto yTest = y.block(testRows.begin, 0, testRows.num, y.cols());

	std::cout << "Train: " << xTrain.rows() << std::endl;
	std::cout << "Val: " << xVal.rows() << std::endl;
	std::cout << "Test: " << xTest.rows() << std::endl;
	std::cout << "Features: " << xTrain.cols() << std::endl;

	std::cout << "Beginning training..." << std::endl;
	Train(nn, epochs, xTrain, yTrain, xVal, yVal, xTest, yTest);
}

// here we have to list all instantiations used (except for in this file)
template void TrainANN<NNMatrixRM, NNVector>(const Eigen::MatrixBase<NNMatrixRM>&, const Eigen::MatrixBase<NNVector>&, EvalNet &, int64_t);
template void TrainANN<NNMatrixRM, NNMatrixRM>(const Eigen::MatrixBase<NNMatrixRM>&, const Eigen::MatrixBase<NNMatrixRM>&, EvalNet &, int64_t);

}
