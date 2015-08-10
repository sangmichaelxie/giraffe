#include "ann_evaluator.h"

#include <fstream>

#include "consts.h"

constexpr float ANNEvaluator::BoundNetErrorAsymmetry;
constexpr float ANNEvaluator::BoundNetTargetShift;

ANNEvaluator::ANNEvaluator()
	: m_evalHash(EvalHashSize)
{
	InvalidateCache();
}

ANNEvaluator::ANNEvaluator(const std::string &filename)
	: m_evalHash(EvalHashSize)
{
	std::ifstream netfIn(filename);
	Deserialize(netfIn);
}

void ANNEvaluator::BuildANN(int64_t inputDims)
{
	m_mainAnn = LearnAnn::BuildEvalNet(inputDims, 1, false);
	m_ubAnn = LearnAnn::BuildEvalNet(inputDims, 1, true);
	m_lbAnn = LearnAnn::BuildEvalNet(inputDims, 1, true);
}

void ANNEvaluator::Serialize(std::ostream &os)
{
	SerializeNet(m_mainAnn, os);
	SerializeNet(m_ubAnn, os);
	SerializeNet(m_lbAnn, os);
}

void ANNEvaluator::Deserialize(std::istream &is)
{
	DeserializeNet(m_mainAnn, is);
	DeserializeNet(m_ubAnn, is);
	DeserializeNet(m_lbAnn, is);

	InvalidateCache();
}

void ANNEvaluator::Train(const std::vector<std::string> &positions, const NNMatrixRM &y, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions, float learningRate)
{
	auto x = BoardsToFeatureRepresentation_(positions, featureDescriptions);

	NNMatrixRM predictions;
	EvalNet::Activations act;

	m_mainAnn.InitializeActivations(act);

	predictions = m_mainAnn.ForwardPropagate(x, act);

	NNMatrixRM errorsDerivative = ComputeErrorDerivatives_(predictions, y, act.actIn[act.actIn.size() - 1], 1.0f, 1.0f);

	EvalNet::Gradients grad;

	m_mainAnn.InitializeGradients(grad);

	m_mainAnn.BackwardPropagateComputeGrad(errorsDerivative, act, grad);

	m_mainAnn.ApplyWeightUpdates(grad, learningRate, 0.0f);

	InvalidateCache();
}

void ANNEvaluator::TrainLoop(const std::vector<std::string> &positions, const NNMatrixRM &y, int64_t epochs, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions)
{
	auto x = BoardsToFeatureRepresentation_(positions, featureDescriptions);

	LearnAnn::TrainANN(x, y, m_mainAnn, epochs);

	InvalidateCache();
}

void ANNEvaluator::TrainBounds(const std::vector<std::string> &positions, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions, float learningRate)
{
	auto x = BoardsToFeatureRepresentation_(positions, featureDescriptions);

	// after training the main net, we train the upper and lower bound nets, using new predictions
	NNMatrixRM newTargets = m_mainAnn.ForwardPropagateFast(x);

	EvalNet::Activations ubAct;
	m_ubAnn.InitializeActivations(ubAct);

	NNMatrixRM ubPredictions = m_ubAnn.ForwardPropagate(x, ubAct);

	NNMatrixRM errorsDerivativeUb = ComputeErrorDerivatives_(ubPredictions, (newTargets.array() + BoundNetTargetShift).matrix(), ubAct.actIn[ubAct.actIn.size() - 1], 1.0f, BoundNetErrorAsymmetry);

	EvalNet::Gradients ubGrad;

	m_ubAnn.InitializeGradients(ubGrad);

	m_ubAnn.BackwardPropagateComputeGrad(errorsDerivativeUb, ubAct, ubGrad);

	m_ubAnn.ApplyWeightUpdates(ubGrad, learningRate, 0.0f);

	EvalNet::Activations lbAct;
	m_lbAnn.InitializeActivations(lbAct);

	NNMatrixRM lbPredictions = m_lbAnn.ForwardPropagate(x, lbAct);

	NNMatrixRM errorsDerivativeLb = ComputeErrorDerivatives_(lbPredictions, (newTargets.array() - BoundNetTargetShift).matrix(), lbAct.actIn[lbAct.actIn.size() - 1], BoundNetErrorAsymmetry, 1.0f);

	EvalNet::Gradients lbGrad;

	m_lbAnn.InitializeGradients(lbGrad);

	m_lbAnn.BackwardPropagateComputeGrad(errorsDerivativeLb, lbAct, lbGrad);

	m_lbAnn.ApplyWeightUpdates(lbGrad, learningRate, 0.0f);

	InvalidateCache();
}

Score ANNEvaluator::EvaluateForWhiteImpl(Board &b, Score /*lowerBound*/, Score /*upperBound*/)
{
	uint64_t hash = b.GetHash();
	EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

	if (entry->hash == hash)
	{
		return entry->val;
	}

	FeaturesConv::ConvertBoardToNN(b, m_convTmp);

	// we have to map every time because the vector's buffer could have moved
	Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

	float annOut = m_mainAnn.ForwardPropagateSingle(mappedVec);

	Score nnRet = annOut * EvalFullScale;

	entry->hash = hash;
	entry->val = nnRet;

	return nnRet;
}

void ANNEvaluator::PrintDiag(Board &board)
{
	FeaturesConv::ConvertBoardToNN(board, m_convTmp);

	Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

	std::cout << "Val: " << m_mainAnn.ForwardPropagateSingle(mappedVec) << std::endl;
	std::cout << "UB: " << m_ubAnn.ForwardPropagateSingle(mappedVec) << std::endl;
	std::cout << "LB: " << m_lbAnn.ForwardPropagateSingle(mappedVec) << std::endl;
}

void ANNEvaluator::InvalidateCache()
{
	for (auto &entry : m_evalHash)
	{
		entry.hash = 0;
	}
}

bool ANNEvaluator::CheckBounds(Board &board, float &windowSize)
{
	FeaturesConv::ConvertBoardToNN(board, m_convTmp);

	Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

	auto exact = m_mainAnn.ForwardPropagateSingle(mappedVec);
	auto ub = m_ubAnn.ForwardPropagateSingle(mappedVec);
	auto lb = m_lbAnn.ForwardPropagateSingle(mappedVec);

	windowSize = fabs(ub - lb);

	return (exact <= ub) && (exact >= lb);
}

NNMatrixRM ANNEvaluator::BoardsToFeatureRepresentation_(const std::vector<std::string> &positions, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions)
{
	NNMatrixRM ret(positions.size(), featureDescriptions.size());

	{
		ScopedThreadLimiter tlim(8);

		#pragma omp parallel
		{
			std::vector<float> features; // each thread reuses a vector to avoid needless allocation/deallocation

			#pragma omp for
			for (size_t i = 0; i < positions.size(); ++i)
			{
				Board b(positions[i]);
				FeaturesConv::ConvertBoardToNN(b, features);

				if (features.size() != featureDescriptions.size())
				{
					std::stringstream msg;

					msg << "Wrong feature vector size! " << features.size() << " (Expecting: " << featureDescriptions.size() << ")";

					throw std::runtime_error(msg.str());
				}

				ret.row(i) = Eigen::Map<NNMatrixRM>(&features[0], 1, static_cast<int64_t>(features.size()));
			}
		}
	}

	return ret;
}

NNMatrixRM ANNEvaluator::ComputeErrorDerivatives_(
	const NNMatrixRM &predictions,
	const NNMatrixRM &targets,
	const NNMatrixRM &finalLayerActivations,
	float positiveWeight,
	float negativeWeight)
{
	// (targets - predictions) * (-1) * dtanh(act)/dz
	int64_t numExamples = predictions.rows();

	NNMatrixRM ret(numExamples, 1);

	// this takes care of everything except the dtanh(act)/dz term, which we can't really vectorize
	ret = (targets - predictions) * -1.0f;

	// derivative of tanh is 1-tanh^2(x)
	for (int64_t i = 0; i < numExamples; ++i)
	{
		float tanhx = tanh(finalLayerActivations(i, 0));
		ret(i, 0) *= 1.0f - tanhx * tanhx;

		if (ret(i, 0) > 0.0f)
		{
			ret(i, 0) *= positiveWeight;
		}
		else
		{
			ret(i, 0) *= negativeWeight;
		}
	}

	return ret;
}
