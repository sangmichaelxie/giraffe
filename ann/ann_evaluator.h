#ifndef ANN_EVALUATOR_H
#define ANN_EVALUATOR_H

#include <vector>
#include <string>

#include <cmath>

#include "evaluator.h"
#include "ann/ann.h"
#include "ann/features_conv.h"
#include "matrix_ops.h"

#include "learn_ann.h"

class ANNEvaluator : public EvaluatorIface
{
public:
	struct EvalHashEntry
	{
		uint64_t hash;
		Score val;
	};

	const static size_t EvalHashSize = 32*MB / sizeof(EvalHashEntry);

	ANNEvaluator();

	ANNEvaluator(const std::string &filename);

	void BuildANN(int64_t inputDims);

	void Serialize(std::ostream &os);

	void Deserialize(std::istream &is);

	void Train(const std::vector<std::string> &positions, const NNMatrixRM &y, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions, float learningRate);

	void TrainLoop(const std::vector<std::string> &positions, const NNMatrixRM &y, int64_t epochs, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions);

	Score EvaluateForWhiteImpl(const Board &b, Score lowerBound, Score upperBound) override;

	void PrintDiag(const std::string &position);

	void InvalidateCache();

private:
	NNMatrixRM BoardsToFeatureRepresentation_(const std::vector<std::string> &positions, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions);

	NNMatrixRM ComputeErrorDerivatives_(
		const NNMatrixRM &predictions,
		const NNMatrixRM &targets,
		const NNMatrixRM &finalLayerActivations);

	EvalNet m_mainAnn;

	std::vector<float> m_convTmp;

	std::vector<EvalHashEntry> m_evalHash;
};

#endif // ANN_EVALUATOR_H
