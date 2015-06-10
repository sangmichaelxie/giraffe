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

#include "ann.h"

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

void ReadMatrixFromFile(const std::string &filename, NNMatrixRM &x)
{
	std::ifstream f(filename, std::ifstream::binary);

	if (!f.is_open())
	{
		throw std::runtime_error(std::string("Failed to open ") + filename + " for reading");
	}

	uint32_t nRows;
	uint32_t nCols;

	f.read(reinterpret_cast<char *>(&nRows), sizeof(uint32_t));
	f.read(reinterpret_cast<char*>(&nCols), sizeof(uint32_t));

	f.seekg(0, std::ios::end);
	size_t fileSize = f.tellg();

	// careful with overflows here
	size_t expectedSize = static_cast<size_t>(nRows) * nCols * sizeof(float) + 2 * sizeof(uint32_t);

	if (fileSize != expectedSize)
	{
		std::cout << filename << " has wrong size!" << std::endl;
		std::cout << "Expected: " << expectedSize << std::endl;
		std::cout << "Actual: " << fileSize << std::endl;
	}

	if ((static_cast<size_t>(nRows) * nCols * sizeof(float)) > MaxMemory)
	{
		nRows = MaxMemory / sizeof(float) / nCols;
	}

	x.resize(nRows, nCols);

	f.seekg(2 * sizeof(uint32_t), std::ios::beg);

	size_t sizeToRead = static_cast<size_t>(nRows) * nCols * sizeof(float);
	size_t sizeRead = 0;

	while (sizeRead != sizeToRead && !f.fail())
	{
		f.read(reinterpret_cast<char *>(x.data()) + sizeRead, std::min<size_t>(8*1024*1024, sizeToRead - sizeRead));
		sizeRead += f.gcount();
	}

	if (f.fail())
	{
		std::cout << "ifstream failed. " << sizeRead << " bytes read." << std::endl;
	}
}

void BuildLayers(const std::string &filename, std::vector<size_t> &layerSizes, std::vector<std::vector<Eigen::Triplet<float> > > &connMatrices)
{
	const size_t FirstHiddenLayerNumGlobalNodes = 1024;
	const size_t FirstHiddenLayerNumNodesPerSquare = 24;

	std::ifstream featuresFile(filename);

	std::vector<size_t> globalFeatures;
	std::vector<size_t> squareConnLists[64]; // list of features influencing each square

	std::string line;

	size_t currentFeatureNum = 0;

	while (std::getline(featuresFile, line))
	{
		std::stringstream ss(line);

		char featureType;
		ss >> featureType;

		if (featureType == 'G')
		{
			globalFeatures.push_back(currentFeatureNum);
		}
		else
		{
			size_t numInfluencedSquares = 0;
			ss >> numInfluencedSquares;

			for (size_t i = 0; i < numInfluencedSquares; ++i)
			{
				int32_t sq;

				ss >> sq;

				squareConnLists[sq].push_back(currentFeatureNum);
			}
		}

		++currentFeatureNum;
	}

	currentFeatureNum = 0;

	// first hidden layer is sparse
	layerSizes.push_back(FirstHiddenLayerNumGlobalNodes + 64 * FirstHiddenLayerNumNodesPerSquare);

	std::vector<Eigen::Triplet<float> > firstHiddenLayerConnMatrix;

	for (size_t i = 0; i < FirstHiddenLayerNumGlobalNodes; ++i)
	{
		for (size_t j = 0; j < globalFeatures.size(); ++j)
		{
			Eigen::Triplet<float> trip(globalFeatures[j], currentFeatureNum, 1.0f);

			firstHiddenLayerConnMatrix.push_back(trip);
		}

		++currentFeatureNum;
	}

	for (size_t sq = 0; sq < 64; ++sq)
	{
		for (size_t i = 0; i < FirstHiddenLayerNumNodesPerSquare; ++i)
		{
			for (size_t j = 0; j < squareConnLists[sq].size(); ++j)
			{
				Eigen::Triplet<float> trip(squareConnLists[sq][j], currentFeatureNum, 1.0f);

				firstHiddenLayerConnMatrix.push_back(trip);
			}

			++currentFeatureNum;
		}
	}

	assert(currentFeatureNum == (FirstHiddenLayerNumGlobalNodes + 64 * FirstHiddenLayerNumNodesPerSquare));

	connMatrices.push_back(firstHiddenLayerConnMatrix);

	// fully connected second layer
	layerSizes.push_back(256);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	// fully connected output layer
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());
}

void SplitDataset(const NNMatrixRM &x, const NNMatrixRM &y, NNMatrixRM &xTrain, NNMatrixRM &yTrain, NNMatrixRM &xVal, NNMatrixRM &yVal, NNMatrixRM &xTest, NNMatrixRM &yTest)
{
	size_t numExamples = x.rows();

	const float testRatio = 0.2f;
	const size_t MaxTest = 5000;
	const float valRatio = 0.2f;
	const size_t MaxVal = 5000;

	size_t testSize = std::min<size_t>(MaxTest, numExamples * testRatio);
	size_t valSize = std::min<size_t>(MaxVal, numExamples * valRatio);
	size_t trainSize = numExamples - testSize - valSize;

	xTest = x.block(0, 0, testSize, x.cols());
	yTest = y.block(0, 0, testSize, y.cols());

	xVal = x.block(testSize, 0, valSize, x.cols());
	yVal = y.block(testSize, 0, valSize, y.cols());

	xTrain = x.block(testSize + valSize, 0, trainSize, x.cols());
	yTrain = y.block(testSize + valSize, 0, trainSize, y.cols());
}

template <typename T>
void Train(T &nn, NNMatrixRM &xTrain, NNMatrixRM &yTrain, NNMatrixRM &xVal, NNMatrixRM &yVal, NNMatrixRM &xTest, NNMatrixRM &yTest, std::mt19937 &mt)
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

void GenerateClusters(const NNMatrixRM &x, NNVector &clusterAssignments, NNMatrixRM &clusterCenters, std::mt19937 &mersenneTwister)
{
	clusterAssignments.resize(1, x.rows());
	clusterCenters.resize(NumClusters, x.cols());

	std::uniform_int_distribution<int64_t> idxDist(0, x.rows() - 1);

	// randomly pick examples to be cluster centers
	for (int64_t i = 0; i < NumClusters; ++i)
	{
		clusterCenters.row(i) = x.row(idxDist(mersenneTwister));
	}

	for (int64_t iteration = 0; iteration < KMeanNumIterations; ++iteration)
	{
		// assign clusters
		for (int64_t i = 0; i < x.rows(); ++i)
		{
			float minNorm = std::numeric_limits<float>::max();
			int64_t minCluster = 0;

			for (int64_t c = 0; c < NumClusters; ++c)
			{
				float norm = (clusterCenters.row(c) - x.row(i)).block(0, 1, 1, 10).lpNorm<1>();
				if (norm < minNorm)
				{
					minNorm = norm;
					minCluster = c;
				}
			}

			clusterAssignments(i) = minCluster;
		}

		// update centers
		clusterCenters.setZero();
		std::vector<int64_t> clusterSizes(NumClusters);

		for (int64_t i = 0; i < x.rows(); ++i)
		{
			int64_t cluster = static_cast<int64_t>(clusterAssignments(i));
			clusterCenters.row(cluster) += x.row(i);
			++clusterSizes[cluster];
		}

		for (int64_t c = 0; c < NumClusters; ++c)
		{
			if (clusterSizes[c] != 0)
			{
				clusterCenters.row(c) /= static_cast<float>(clusterSizes[c]);
			}
		}

		for (size_t c = 0; c < NumClusters; ++c)
		{
			std::cout << clusterSizes[c] << " ";
		}
		std::cout << std::endl;
	}
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

template <typename T>
void PrintTestStats(T &nn, NNMatrixRM &x, NNMatrixRM &y)
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

	std::mt19937 mersenneTwister(42);

	std::vector<size_t> hiddenLayersConfig;
	std::vector<std::vector<Eigen::Triplet<float> > > connMatrices;

	BuildLayers(featuresFilename, hiddenLayersConfig, connMatrices);

	NNMatrix xClusters[NumClusters];
	NNMatrix yClusters[NumClusters];

	NNMatrixRM xTrain, yTrain, xVal, yVal, xTest, yTest;

	{
		// this inner scope is to release memory for x and y before training starts
		NNMatrixRM x;
		NNMatrixRM y;

		ReadMatrixFromFile(xFilename, x);
		ReadMatrixFromFile(yFilename, y);
		size_t clusterIdx[NumClusters]; // current indices

		NNVector clusterAssignments;
		NNMatrix clusterCenters;
		GenerateClusters(x, clusterAssignments, clusterCenters, mersenneTwister);

		std::vector<int64_t> clusterSizes(NumClusters);
		for (int64_t i = 0; i < x.rows(); ++i)
		{
			++clusterSizes[static_cast<int64_t>(clusterAssignments(i))];
		}

		for (int64_t c = 0; c < NumClusters; ++c)
		{
			xClusters[c].resize(clusterSizes[c], x.cols());
			yClusters[c].resize(clusterSizes[c], y.cols());
			clusterIdx[c] = 0;
		}

		for (int64_t i = 0; i < x.rows(); ++i)
		{
			int64_t cluster = static_cast<int64_t>(clusterAssignments(i));
			xClusters[cluster].row(clusterIdx[cluster]) = x.row(i);
			yClusters[cluster].row(clusterIdx[cluster]) = y.row(i);
			++clusterIdx[cluster];
		}
	}

	int64_t threadCount = 1;
	const char *threadCountEnv = std::getenv("NUM_THREADS");
	if (threadCountEnv)
	{
		threadCount = atoi(threadCountEnv);
	}

	std::cout << "Using " << threadCount << " threads" << std::endl;

	for (int64_t c = 0; c < NumClusters; ++c)
	{
		SplitDataset(xClusters[c], yClusters[c], xTrain, yTrain, xVal, yVal, xTest, yTest);

		std::cout << "Train: " << xTrain.rows() << std::endl;
		std::cout << "Val: " << xVal.rows() << std::endl;
		std::cout << "Test: " << xTest.rows() << std::endl;
		std::cout << "Features: " << xTrain.cols() << std::endl;

		FCANN<Relu> nn(77, xTrain.cols(), 1, hiddenLayersConfig, connMatrices);

		nn.SetNumThreads(threadCount);

		Train(nn, xTrain, yTrain, xVal, yVal, xTest, yTest, mersenneTwister);

		// compute test performance and statistics
		PrintTestStats(nn, xTest, yTest);
	}
}
}
