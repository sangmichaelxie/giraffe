#include "learn.h"

#include <stdexcept>
#include <vector>
#include <sstream>

#include <omp.h>

#include "matrix_ops.h"
#include "board.h"
#include "ann/features_conv.h"

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

	std::vector<std::string> trainingPositions;

	std::string fen;

	size_t fenMemUsage = 0;

	std::cout << "Reading FENs..." << std::endl;

	while (std::getline(positionsFile, fen))
	{
		trainingPositions.push_back(fen);

		fenMemUsage += fen.size();
	}

	std::cout << "Positions read: " << trainingPositions.size() << std::endl;
	std::cout << "Approx mem usage for FENs: " << (fenMemUsage / 1024 / 1024) << " MB" << std::endl;

	std::cout << "Converting boards to features..." << std::endl;

	std::vector<FeaturesConv::FeatureDescription> featureDescriptions =
		FeaturesConv::ConvertBoardToNN<FeaturesConv::FeatureDescription>(Board());

	NNMatrixRM boardsInFeatureRepresentation(static_cast<int64_t>(trainingPositions.size()), static_cast<int64_t>(featureDescriptions.size()));

	#pragma omp parallel for
	for (size_t i = 0; i < trainingPositions.size(); ++i)
	{
		std::vector<float> features = FeaturesConv::ConvertBoardToNN<float>(Board(trainingPositions[i]));

		if (features.size() != featureDescriptions.size())
		{
			std::stringstream msg;

			msg << "Wrong feature vector size! " << features.size() << " (Expecting: " << featureDescriptions.size() << ")";

			throw std::runtime_error(msg.str());
		}

		boardsInFeatureRepresentation.row(i) = Eigen::Map<NNVector>(&features[0], 1, static_cast<int64_t>(features.size()));
	}

	std::cout << "Memory usage for boards in feature representation: " <<
		(sizeof(float) * featureDescriptions.size() * trainingPositions.size() / 1024 / 1024) << " MB" << std::endl;

	NNVector trainingTargets(trainingPositions.size());

	for (int64_t iter = 0; iter < NumIterations; ++iter)
	{

	}
}

}
