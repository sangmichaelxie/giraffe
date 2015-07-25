#include "ann_evaluator.h"

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
	m_mainAnn = LearnAnn::BuildNet<EvalNet>(inputDims, 1);
}

void ANNEvaluator::Serialize(std::ostream &os)
{
	SerializeNet(m_mainAnn, os);
}

void ANNEvaluator::Deserialize(std::istream &is)
{
	DeserializeNet(m_mainAnn, is);

	InvalidateCache();
}

void ANNEvaluator::Train(const std::vector<std::string> &positions, const NNMatrixRM &y, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions, float learningRate)
{
	auto x = BoardsToFeatureRepresentation_(positions, featureDescriptions);

	NNMatrixRM predictions;
	EvalNet::Activations act;

	m_mainAnn.InitializeActivations(act);

	predictions = m_mainAnn.ForwardPropagate(x, act);

	// we are using MSE with linear output, so derivative is just diff
	NNMatrixRM errorsDerivative = ComputeErrorDerivatives_(predictions, y, act.actIn[act.actIn.size() - 1]);

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

Score ANNEvaluator::EvaluateForWhiteImpl(const Board &b, Score /*lowerBound*/, Score /*upperBound*/)
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

void ANNEvaluator::PrintDiag(const std::string &position)
{
	std::cout << position << std::endl;

	FeaturesConv::ConvertBoardToNN(Board(position), m_convTmp);

	Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

	std::cout << "Val: " << m_mainAnn.ForwardPropagateSingle(mappedVec) << std::endl;
}

void ANNEvaluator::InvalidateCache()
{
	for (auto &entry : m_evalHash)
	{
		entry.hash = 0;
	}
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
				FeaturesConv::ConvertBoardToNN(Board(positions[i]), features);

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
	const NNMatrixRM &finalLayerActivations)
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
	}

	return ret;
}
