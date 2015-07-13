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
#include <omp.h>

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

namespace
{
const int64_t KMeanNumIterations = 1;

const size_t MaxBatchSize = 256;

const size_t MaxMemory = 32ULL*1024*1024*1024; // limit dataset size if we have many features

const size_t MaxIterationsPerCheck = 500000 / MaxBatchSize;

const float ExclusionFactor = 0.99f; // when computing test performance, ignore 1% of outliers

class MMappedMatrix
{
public:
	MMappedMatrix(const std::string &filename)
	{
#ifndef _WIN32
		m_fd = open(filename.c_str(), O_RDONLY);

		if (m_fd == -1)
		{
			throw std::runtime_error(std::string("Failed to open ") + filename + " for reading");
		}

		// we first map the first 8 bytes to read the size of the matrix, before doing the actual mapping
		m_mappedAddress = mmap(0, 8, PROT_READ, MAP_SHARED, m_fd, 0);

		if (m_mappedAddress == MAP_FAILED)
		{
			throw std::runtime_error("Failed to map file for reading matrix size");
		}

		// read the size of the matrix
		m_rows = *(reinterpret_cast<uint32_t *>(m_mappedAddress));
		m_cols = *(reinterpret_cast<uint32_t *>(m_mappedAddress) + 1);

		munmap(m_mappedAddress, 8);

		m_mapSize = static_cast<size_t>(m_rows) * m_cols * sizeof(float) + 8;

		// map again using the actual size (we can't use the offset here because it
		// must be a multiple of page size)
		m_mappedAddress = mmap(0, m_mapSize, PROT_READ, MAP_SHARED, m_fd, 0);

		if (m_mappedAddress == MAP_FAILED)
		{
			throw std::runtime_error("Failed to map file for reading");
		}

		madvise(m_mappedAddress, m_mapSize, MADV_SEQUENTIAL);

		m_matrixStartAddress = reinterpret_cast<float*>(reinterpret_cast<char*>(m_mappedAddress) + 8);
#else
		assert(false && "MMappedMatrix not implemented on Windows!");
#endif
	}

	Eigen::Map<NNMatrixRM> GetMap() { return Eigen::Map<NNMatrixRM>(m_matrixStartAddress, m_rows, m_cols); }

	MMappedMatrix(const MMappedMatrix &) = delete;
	MMappedMatrix &operator=(const MMappedMatrix &) = delete;

	~MMappedMatrix()
	{
#ifndef _WIN32
		munmap(m_mappedAddress, m_mapSize);
#endif
	}

private:
	int m_fd;
	void *m_mappedAddress;
	float *m_matrixStartAddress;

	size_t m_mapSize;

	uint32_t m_rows;
	uint32_t m_cols;
};

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

struct LayerDescription
{
	size_t layerSize;
	std::vector<Eigen::Triplet<float> > connections;
	std::vector<std::vector<int32_t>> groups;
};

LayerDescription BuildLocalLayer(const std::vector<std::vector<int32_t>> &groupsIn, float nodeRatio, size_t maxNodesPerGroup)
{
	LayerDescription ret;

	ret.groups.resize(groupsIn.size());

	size_t groupOut = 0;
	size_t node = 0;

	for (auto group : groupsIn)
	{
		size_t nodesInGroup = group.size();
		size_t nodesForThisGroup = std::min<size_t>(nodesInGroup * nodeRatio, maxNodesPerGroup);

		for (size_t i = 0; i < nodesForThisGroup; ++i)
		{
			for (auto feature : group)
			{
				ret.connections.push_back(Eigen::Triplet<float>(feature, node, 1.0f));
			}

			ret.groups[groupOut].push_back(node);

			++node;
		}

		++groupOut;
	}

	ret.layerSize = node;

	return ret;
}

} // namespace

namespace LearnAnn
{

template <typename T>
T BuildNet(int64_t inputDims, int64_t outputDims)
{
	std::vector<size_t> layerSizes;
	std::vector<std::vector<Eigen::Triplet<float> > > connMatrices;

	std::vector<std::vector<int32_t>> featureGroups;

	// get feature descriptions
	std::vector<FeaturesConv::FeatureDescription> featureDescriptions;
	FeaturesConv::ConvertBoardToNN(Board(), featureDescriptions);

	// first we make global feature groups
	std::map<int, std::vector<int32_t>> globalGroups;
	for (size_t featureNum = 0; featureNum < featureDescriptions.size(); ++featureNum)
	{
		auto &fd = featureDescriptions[featureNum];

		if (fd.featureType == FeaturesConv::FeatureDescription::FeatureType_global)
		{
			globalGroups[fd.group].push_back(featureNum);
		}
	}

	for (const auto &group : globalGroups)
	{
		featureGroups.push_back(group.second);
	}

	// now we make square-specific groups
	std::vector<std::vector<int32_t>> xGroups(8);
	std::vector<std::vector<int32_t>> yGroups(8);
	std::vector<std::vector<int32_t>> diag0Groups(15);
	std::vector<std::vector<int32_t>> diag1Groups(15);

	for (size_t featureNum = 0; featureNum < featureDescriptions.size(); ++featureNum)
	{
		auto &fd = featureDescriptions[featureNum];

		if (fd.featureType == FeaturesConv::FeatureDescription::FeatureType_pos)
		{
			Square sq = fd.sq;

			int32_t x = GetX(sq);
			int32_t y = GetY(sq);
			int32_t diag0 = GetDiag0(sq);
			int32_t diag1 = GetDiag1(sq);

			xGroups[x].push_back(featureNum);
			yGroups[y].push_back(featureNum);
			diag0Groups[diag0].push_back(featureNum);
			diag1Groups[diag1].push_back(featureNum);
		}
	}

	//featureGroups.insert(featureGroups.end(), xGroups.begin(), xGroups.end());
	//featureGroups.insert(featureGroups.end(), yGroups.begin(), yGroups.end());
	//featureGroups.insert(featureGroups.end(), diag0Groups.begin(), diag0Groups.end());
	//featureGroups.insert(featureGroups.end(), diag1Groups.begin(), diag1Groups.end());

	LayerDescription layer0 = BuildLocalLayer(featureGroups, 1.0f, 16);

	layerSizes.push_back(layer0.layerSize);
	connMatrices.push_back(layer0.connections);

	LayerDescription layer1 = BuildLocalLayer(layer0.groups, 1.0f, 16);

	layerSizes.push_back(layer1.layerSize);
	connMatrices.push_back(layer1.connections);

	layerSizes.push_back(512);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	layerSizes.push_back(64);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	// fully connected output layer
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	return T(inputDims, outputDims, layerSizes, connMatrices);
}

template EvalNet BuildNet(int64_t inputDims, int64_t outputDims);
template MixingNet BuildNet(int64_t inputDims, int64_t outputDims);

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

}
