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
#include <sys/mman.h>
#include <fcntl.h>

#include "ann.h"
#include "types.h"

namespace
{
const int64_t NumClusters = 1;
const int64_t KMeanNumIterations = 1;

const size_t BatchSize = 256;

const size_t MaxMemory = 32ULL*1024*1024*1024; // limit dataset size if we have many features

const size_t IterationsPerCheck = 500000 / BatchSize;

// how many examples to see (this is 30 epochs for 5M examples)
const int64_t ExamplesLimit = 150000000LL;

const float ExclusionFactor = 0.99f; // when computing test performance, ignore 1% of outliers

class MMappedMatrix
{
public:
	MMappedMatrix(const std::string &filename)
	{
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
	}

	Eigen::Map<NNMatrixRM> GetMap() { return Eigen::Map<NNMatrixRM>(m_matrixStartAddress, m_rows, m_cols); }

	MMappedMatrix(const MMappedMatrix &) = delete;
	MMappedMatrix &operator=(const MMappedMatrix &) = delete;

	~MMappedMatrix() { munmap(m_mappedAddress, m_mapSize); }

private:
	int m_fd;
	void *m_mappedAddress;
	float *m_matrixStartAddress;

	size_t m_mapSize;

	uint32_t m_rows;
	uint32_t m_cols;
};

void BuildLayers(const std::string &filename, std::vector<size_t> &layerSizes, std::vector<std::vector<Eigen::Triplet<float> > > &connMatrices)
{
	std::vector<std::vector<int32_t> > featureGroups;

	std::ifstream featureFile(filename);

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

	// in the first layer, we connect the groups locally
	// if a group has x features, we use x*FeatureCountMultiplier hidden nodes
	float FeatureCountMultiplier = 2.0f;
	size_t nodeCount = 0;

	std::vector<Eigen::Triplet<float> > connections;
	for (size_t group = 0; group < featureGroups.size(); ++group)
	{
		size_t numNodes = featureGroups[group].size() * FeatureCountMultiplier;

		for (size_t node = 0; node < numNodes; ++node)
		{
			// connect the node to all nodes in the group
			for (const auto &feature : featureGroups[group])
			{
				connections.push_back(Eigen::Triplet<float>(feature, nodeCount, 1.0f));
			}

			++nodeCount;
		}
	}

	std::cout << connections.size() << " " << (static_cast<int64_t>(nodeCount) * 223) << std::endl;

	layerSizes.push_back(nodeCount);
	connMatrices.push_back(connections);

	layerSizes.push_back(1024);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	layerSizes.push_back(256);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	// fully connected output layer
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());
}

struct Rows
{
	Rows() {}
	Rows(int64_t begin, int64_t num) : begin(begin), num(num) {}

	int64_t begin;
	int64_t num;
};

template <typename Derived>
void SplitDataset(
	Eigen::MatrixBase<Derived> &x,
	Eigen::MatrixBase<Derived> &y,
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

template <typename T, typename Derived>
void Train(
	T &nn,
	Eigen::MatrixBase<Derived> &xTrain,
	Eigen::MatrixBase<Derived> &yTrain,
	Eigen::MatrixBase<Derived> &xVal,
	Eigen::MatrixBase<Derived> &yVal,
	Eigen::MatrixBase<Derived> &xTest,
	Eigen::MatrixBase<Derived> &yTest,
	std::mt19937 &mt)
{
	size_t iter = 0;

	float trainingErrorAccum = 0.0f;

	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	T bestNet = nn;
	FP bestValScore = std::numeric_limits<FP>::max(); // this is updated every time val score improves

	bool done = false;

	size_t NumBatches = xTrain.rows() / BatchSize;

	int64_t epoch = 0;

	while (!done && (iter * BatchSize) < ExamplesLimit)
	{
		size_t batchNum = iter % NumBatches;

		size_t begin = batchNum * BatchSize;

		epoch = iter * BatchSize / xTrain.rows();

		trainingErrorAccum += nn.TrainGDM(
			xTrain.block(begin, 0, BatchSize, xTrain.cols()),
			yTrain.block(begin, 0, BatchSize, 1),
			0.000001f);

		if (iter % IterationsPerCheck == 0)
		{
			NNMatrix pred = nn.ForwardPropagateFast(xVal);

			NNMatrix eVal = pred - yVal;
			NNMatrix errors = eVal;

			ErrorFunc(eVal, errors);

			FP valScore = errors.sum() / xVal.rows();

			if (valScore < bestValScore)
			{
				bestValScore = valScore;
				bestNet = nn;
			}

			std::cout << "Iteration: " << iter << ", ";

			std::cout << "Epoch: " << epoch << ", ";

			std::cout << "Val: " << valScore << ", ";

			std::cout << "Train: " << (trainingErrorAccum / std::min(iter + 1, IterationsPerCheck)) << ", ";

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
NNMatrix EvalNet(T &nn, NNMatrixRM &x)
{
	// how many examples to evaluate at a time (memory restriction)
	const static int64_t ExamplesPerBatch = 2048;

	NNMatrix ret(x.rows(), 1);

	for (int64_t i = 0; i < x.rows();)
	{
		int64_t examplesToEval = std::min<int64_t>(x.rows() - i, ExamplesPerBatch);

		NNMatrix pred = nn.ForwardPropagateFast(x.block(i, 0, examplesToEval, x.cols()));

		ret.block(i, 0, examplesToEval, ret.cols()) = pred;

		i += examplesToEval;
	}

	return ret;
}

template <typename T, typename Derived>
void PrintTestStats(T &nn, Eigen::MatrixBase<Derived> &x, Eigen::MatrixBase<Derived> &y)
{
	NNMatrix pred = nn.ForwardPropagateFast(x);

	NNMatrix eDiff = pred - y;
	NNMatrix errors = eDiff; // create a matrix of the same size

	ErrorFunc(eDiff, errors);

	std::cout << "\n\nStatistics:" << std::endl;

	std::vector<float> errorsVec;

	for (int64_t i = 0; i < x.rows(); ++i)
	{
		errorsVec.push_back(errors(i, 0));
	}

	std::sort(errorsVec.begin(), errorsVec.end());

	float score = 0;

	for (size_t i = 0; i < (errorsVec.size() * ExclusionFactor); ++i)
	{
		score += errorsVec[i];
	}

	score /= (errorsVec.size() * ExclusionFactor);

	std::cout << "Test perf (EF: " << ExclusionFactor << "): " << score << std::endl;

	const float Bins[] = { 5.0f, 10.0f, 15.0f, 20.0f, 35.0f, 50.0f, 75.0f, 100.0f, 150.0f, 200.0f, 400.0f, 1000.0f, 0.0f /* catch all bin */ };
	const size_t NumBins = sizeof(Bins) / sizeof(Bins[0]);
	size_t binCounts[NumBins] = { 0 };

	for (int64_t i = 0; i < x.rows(); ++i)
	{
		float e = errors(i, 0);

		size_t bin = NumBins - 1;

		for (size_t b = 0; b < NumBins; ++b)
		{
			if (e <= Bins[b])
			{
				bin = b;
				break;
			}
		}

		++binCounts[bin];
	}

	size_t cumulativeCount = 0;

	for (size_t b = 0; b < NumBins; ++b)
	{
		if (b != (NumBins - 1))
		{
			std::cout << "<" << Bins[b] << ": ";
		}
		else
		{
			std::cout << ">=" << Bins[b-1] << ": ";
		}

		cumulativeCount += binCounts[b];

		std::cout << binCounts[b] << " (" << ((100.0f * cumulativeCount) / errorsVec.size()) << "%)" <<  std::endl;
	}
}
} // namespace

namespace LearnAnn
{
void Learn(
	const std::string &xFilename,
	const std::string &yFilename,
	const std::string &featuresFilename)
{
	EnableNanInterrupt();

	// if we are using GPU, disable OMP (our own parallelization)
	// and use Eigen's instead
	// our implementation has lower overhead, but requires slicing matrices
#ifdef VIENNACL_WITH_OPENCL
	Eigen::setNbThreads(omp_get_max_threads());
	omp_set_num_threads(1);
#endif

	std::mt19937 mersenneTwister(42);

	std::vector<size_t> hiddenLayersConfig;
	std::vector<std::vector<Eigen::Triplet<float> > > connMatrices;

	BuildLayers(featuresFilename, hiddenLayersConfig, connMatrices);

	MMappedMatrix xMap(xFilename);
	MMappedMatrix yMap(yFilename);

	auto x = xMap.GetMap();
	auto y = yMap.GetMap();

	std::cout << "Using " << omp_get_max_threads() << " thread(s)" << std::endl;

	for (int64_t c = 0; c < NumClusters; ++c)
	{
		Rows trainRows, valRows, testRows;
		SplitDataset(x, y, trainRows, valRows, testRows);

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

		FCANN<Relu> nn(77, xTrain.cols(), 1, hiddenLayersConfig, connMatrices);

		//std::ifstream netfIn("net.dump");
		//FCANN<Relu> nn = DeserializeNet<Relu>(netfIn);

		std::cout << "Beginning training..." << std::endl;
		Train(nn, xTrain, yTrain, xVal, yVal, xTest, yTest, mersenneTwister);

		// compute test performance and statistics
		PrintTestStats(nn, xTest, yTest);

		//std::ofstream netf("net.dump");
		//SerializeNet(nn, netf);
	}
}
}
