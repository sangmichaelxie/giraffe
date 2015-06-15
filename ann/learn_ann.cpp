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
	const size_t FirstHiddenLayerNumGlobalNodes = 256;
	const size_t FirstHiddenLayerNumNodesPerSquare = 16;

	const size_t SecondHiddenLayerNumGlobalNodes = 64;
	const size_t SecondHiddenLayerNumNodesPerSquare = 4;

	std::ifstream featuresFile(filename);

	std::vector<size_t> firstHiddenLayerGlobalFeatures;
	std::vector<size_t> firstHiddenLayerSquareConnLists[64]; // list of features influencing each square for first hidden layer
	std::vector<size_t> secondHiddenLayerGlobalFeatures; // list of global features from first hidden layer output
	std::vector<size_t> secondHiddenLayerSquareConnLists[64]; // list of features influencing each square for second hidden layer

	std::string line;

	size_t currentFeatureNum = 0;

	// for the first hidden layer, we get the influences from feature file
	while (std::getline(featuresFile, line))
	{
		std::stringstream ss(line);

		char featureType;
		ss >> featureType;

		if (featureType == 'G')
		{
			firstHiddenLayerGlobalFeatures.push_back(currentFeatureNum);
		}
		else
		{
			size_t numInfluencedSquares = 0;
			ss >> numInfluencedSquares;

			for (size_t i = 0; i < numInfluencedSquares; ++i)
			{
				int32_t sq;

				ss >> sq;

				firstHiddenLayerSquareConnLists[sq].push_back(currentFeatureNum);
			}
		}

		++currentFeatureNum;
	}

	currentFeatureNum = 0;

	// first hidden layer is sparse
	std::vector<Eigen::Triplet<float> > firstHiddenLayerConnMatrix;

	for (size_t i = 0; i < FirstHiddenLayerNumGlobalNodes; ++i)
	{
		for (size_t j = 0; j < firstHiddenLayerGlobalFeatures.size(); ++j)
		{
			Eigen::Triplet<float> trip(firstHiddenLayerGlobalFeatures[j], currentFeatureNum, 1.0f);

			firstHiddenLayerConnMatrix.push_back(trip);
		}

		secondHiddenLayerGlobalFeatures.push_back(currentFeatureNum);

		++currentFeatureNum;
	}

	for (size_t sq = 0; sq < 64; ++sq)
	{
		for (size_t i = 0; i < FirstHiddenLayerNumNodesPerSquare; ++i)
		{
			for (size_t j = 0; j < firstHiddenLayerSquareConnLists[sq].size(); ++j)
			{
				Eigen::Triplet<float> trip(firstHiddenLayerSquareConnLists[sq][j], currentFeatureNum, 1.0f);

				firstHiddenLayerConnMatrix.push_back(trip);
			}

			// this square affects adjacent squares
			// now we build the connection list for second layer
			int32_t x = GetX(sq);
			int32_t y = GetY(sq);
			for (int32_t offsetX = -1; offsetX <= 1; ++offsetX)
			{
				for (int32_t offsetY = -1; offsetY <= 1; ++offsetY)
				{
					if (Valid(x + offsetX) && Valid(y + offsetY))
					{
						secondHiddenLayerSquareConnLists[Sq(x + offsetX, y + offsetY)].push_back(currentFeatureNum);
					}
				}
			}

			++currentFeatureNum;
		}
	}

	assert(currentFeatureNum == (FirstHiddenLayerNumGlobalNodes + 64 * FirstHiddenLayerNumNodesPerSquare));
	layerSizes.push_back(FirstHiddenLayerNumGlobalNodes + 64 * FirstHiddenLayerNumNodesPerSquare);
	connMatrices.push_back(firstHiddenLayerConnMatrix);
	currentFeatureNum = 0;

	// second hidden layer is sparse
	std::vector<Eigen::Triplet<float> > secondHiddenLayerConnMatrix;

	for (size_t i = 0; i < SecondHiddenLayerNumGlobalNodes; ++i)
	{
		for (size_t j = 0; j < secondHiddenLayerGlobalFeatures.size(); ++j)
		{
			Eigen::Triplet<float> trip(secondHiddenLayerGlobalFeatures[j], currentFeatureNum, 1.0f);

			secondHiddenLayerConnMatrix.push_back(trip);
		}

		++currentFeatureNum;
	}

	for (size_t sq = 0; sq < 64; ++sq)
	{
		for (size_t i = 0; i < SecondHiddenLayerNumNodesPerSquare; ++i)
		{
			for (size_t j = 0; j < secondHiddenLayerSquareConnLists[sq].size(); ++j)
			{
				Eigen::Triplet<float> trip(secondHiddenLayerSquareConnLists[sq][j], currentFeatureNum, 1.0f);

				secondHiddenLayerConnMatrix.push_back(trip);
			}

			++currentFeatureNum;
		}
	}

	assert(currentFeatureNum == (SecondHiddenLayerNumGlobalNodes + 64 * SecondHiddenLayerNumNodesPerSquare));
	//layerSizes.push_back(SecondHiddenLayerNumGlobalNodes + 64 * SecondHiddenLayerNumNodesPerSquare);
	//connMatrices.push_back(secondHiddenLayerConnMatrix);

	// fully connected third layer
	//layerSizes.push_back(128);
	//connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

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

		std::cout << "Beginning training..." << std::endl;
		Train(nn, xTrain, yTrain, xVal, yVal, xTest, yTest, mersenneTwister);

		// compute test performance and statistics
		PrintTestStats(nn, xTest, yTest);
	}
}
}
