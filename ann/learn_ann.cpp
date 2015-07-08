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

} // namespace

namespace LearnAnn
{

EvalNet BuildEvalNet(const std::string &featureFilename, int64_t inputDims, std::mt19937 &mt)
{
	std::vector<size_t> layerSizes;
	std::vector<std::vector<Eigen::Triplet<float> > > connMatrices;

	std::vector<std::vector<int32_t> > featureGroups;

	std::ifstream featureFile(featureFilename);

	char type;

	int32_t feature = 0;

	while (featureFile >> type)
	{
		int32_t group;
		featureFile >> group;

		if (static_cast<size_t>(group) >= featureGroups.size())
		{
			featureGroups.resize(group + 1);
		}

		featureGroups[group].push_back(feature);

		++feature;
	}

	std::vector<Eigen::Triplet<float> > connections;

	// build first layer
	const size_t FirstHiddenLayerNodes = 512;
	const size_t FirstHiddenLayerNumGroupsPerNode = 4;
	connections.clear();

	std::uniform_int_distribution<> groupDist(0, featureGroups.size() - 1);

	for (size_t node = 0; node < FirstHiddenLayerNodes; ++node)
	{
		for (size_t i = 0; i < FirstHiddenLayerNumGroupsPerNode; ++i)
		{
			// we can have duplicates, in which case we'll actually have less than 4 groups, and that's ok
			size_t group = groupDist(mt);

			// connect the node to all nodes in the group
			for (const auto &feature : featureGroups[group])
			{
				connections.push_back(Eigen::Triplet<float>(feature, node, 1.0f));
			}
		}
	}

	layerSizes.push_back(FirstHiddenLayerNodes);
	connMatrices.push_back(connections);

	layerSizes.push_back(256);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	layerSizes.push_back(64);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	// fully connected output layer
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	return EvalNet(mt(), inputDims, 1, layerSizes, connMatrices);
}

template <typename Derived1, typename Derived2>
EvalNet TrainANN(
	const Eigen::MatrixBase<Derived1> &x,
	const Eigen::MatrixBase<Derived2> &y,
	const std::string &featuresFilename,
	EvalNet *start,
	int64_t epochs)
{
	std::mt19937 mersenneTwister(42);

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

	EvalNet nn = BuildEvalNet(featuresFilename, xTrain.cols(), mersenneTwister);

	if (start)
	{
		nn = *start;
	}

	std::cout << "Beginning training..." << std::endl;
	Train(nn, epochs, xTrain, yTrain, xVal, yVal, xTest, yTest);

	return nn;
}

// here we have to list all instantiations used (except for in this file)
template EvalNet TrainANN<NNMatrixRM, NNVector>(const Eigen::MatrixBase<NNMatrixRM>&, const Eigen::MatrixBase<NNVector>&, const std::string&, EvalNet *, int64_t);

EvalNet TrainANNFromFile(
	const std::string &xFilename,
	const std::string &yFilename,
	const std::string &featuresFilename,
	EvalNet *start,
	int64_t epochs)
{
	MMappedMatrix xMap(xFilename);
	MMappedMatrix yMap(yFilename);

	auto x = xMap.GetMap();
	auto y = yMap.GetMap();

	return TrainANN(x, y, featuresFilename, start, epochs);
}
}
