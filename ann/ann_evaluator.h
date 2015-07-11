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

	const static size_t EvalHashSize = 1*MB / sizeof(EvalHashEntry);

	const static size_t NumNets = 1;

	ANNEvaluator() : m_anns(NumNets), m_evalHash(EvalHashSize) { InvalidateCache(); }

	ANNEvaluator(const std::vector<EvalNet> &anns) : m_anns(anns), m_evalHash(EvalHashSize)	{ InvalidateCache(); }

	ANNEvaluator(const std::string &filename) : m_evalHash(EvalHashSize)
	{
		std::ifstream netfIn(filename);

		Deserialize(netfIn);
	}

	void BuildANN(const std::string &featureFilename, int64_t inputDims)
	{
		m_anns.resize(NumNets);

		for (auto &net : m_anns)
		{
			net = LearnAnn::BuildEvalNet(featureFilename, inputDims);
		}

		m_mixingNet = LearnAnn::BuildMixingNet(featureFilename, inputDims, m_anns.size());
	}

	void Serialize(std::ostream &os)
	{
		SerializeNet(m_mixingNet, os);

		os << m_anns.size() << std::endl;

		for (auto &net : m_anns)
		{
			SerializeNet(net, os);
		}
	}

	void Deserialize(std::istream &is)
	{
		DeserializeNet(m_mixingNet, is);

		size_t n;

		is >> n;

		m_anns.resize(n);

		for (auto &net : m_anns)
		{
			DeserializeNet(net, is);
		}

		InvalidateCache();
	}

	void Train(const std::vector<std::string> &positions, const NNMatrixRM &y, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions)
	{
		auto x = BoardsToFeatureRepresentation_(positions, featureDescriptions);

		MixingNet::Activations mixingAct;

		m_mixingNet.InitializeActivations(mixingAct);

		auto sampleWeights = m_mixingNet.ForwardPropagate(x, mixingAct);

		std::vector<NNMatrixRM> predictions(m_anns.size());
		std::vector<EvalNet::Activations> acts(m_anns.size());

		for (size_t i = 0; i < m_anns.size(); ++i)
		{
			m_anns[i].InitializeActivations(acts[i]);

			predictions[i] = m_anns[i].ForwardPropagate(x, acts[i]);
		}

		NNMatrixRM predictionsCombined = NNMatrixRM::Zero(predictions[0].rows(), predictions[0].cols());

		for (size_t i = 0; i < m_anns.size(); ++i)
		{
			predictionsCombined += (predictions[i].array() * sampleWeights.col(i).array()).matrix();
		}

		for (size_t i = 0; i < m_anns.size(); ++i)
		{
			NNMatrixRM errorsDerivative = ComputeExpertErrorDerivatives_(predictionsCombined, y, sampleWeights, i, acts[i].actIn[acts[i].actIn.size() - 1]);

			EvalNet::Gradients grad;

			m_anns[i].InitializeGradients(grad);

			m_anns[i].BackwardPropagateComputeGrad(errorsDerivative, acts[i], grad);

			m_anns[i].ApplyWeightUpdates(grad, 0.0f);
		}

		// now train the mixing net
		NNMatrixRM mixingNetErrorDerivative = ComputeMixingNetErrorDerivatives_(predictionsCombined, predictions, y, sampleWeights);

		MixingNet::Gradients mixingGrad;
		m_mixingNet.InitializeGradients(mixingGrad);

		m_mixingNet.BackwardPropagateComputeGrad(mixingNetErrorDerivative, mixingAct, mixingGrad);

		m_mixingNet.ApplyWeightUpdates(mixingGrad, 0.0f);

		InvalidateCache();
	}

	void TrainLoop(const std::vector<std::string> &positions, const NNMatrixRM &y, int64_t epochs, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions)
	{
		auto x = BoardsToFeatureRepresentation_(positions, featureDescriptions);

		for (size_t i = 0; i < m_anns.size(); ++i)
		{
			LearnAnn::TrainANN(x, y, m_anns[i], epochs);
		}

		InvalidateCache();
	}

	Score EvaluateForWhiteImpl(const Board &b, Score /*lowerBound*/, Score /*upperBound*/) override
	{
		uint64_t hash = b.GetHash();
		EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

		if (entry->hash == hash)
		{
			return entry->val;
		}

		FeaturesConv::ConvertBoardToNN(b, m_convTmp);

		Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

		auto sampleWeights = ComputeSampleWeights_(mappedVec);

		float sum = 0.0f;

		assert(sampleWeights.cols() == static_cast<int64_t>(m_anns.size()));
		assert(sampleWeights.rows() == 1);
		assert(sampleWeights.sum() > 0.9999f && sampleWeights.sum() < 1.0001f);

		for (int64_t i = 0; i < static_cast<int64_t>(m_anns.size()); ++i)
		{
			sum += m_anns[i].ForwardPropagateSingle(mappedVec) * sampleWeights(0, i);
		}

		Score nnRet = sum * EvalFullScale;

		entry->hash = hash;
		entry->val = nnRet;

		return nnRet;
	}

	void PrintDiag(const std::string &position)
	{
		std::cout << position << std::endl;

		FeaturesConv::ConvertBoardToNN(Board(position), m_convTmp);

		Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

		auto weights = ComputeSampleWeights_(mappedVec);

		std::cout << "Weights: ";

		for (size_t i = 0; i < m_anns.size(); ++i)
		{
			std::cout << '\t' << weights(0, i) << ' ';
		}
		std::cout << std::endl;

		std::cout << "Vals: \t";

		for (size_t i = 0; i < m_anns.size(); ++i)
		{
			std::cout << '\t' << m_anns[i].ForwardPropagateSingle(mappedVec) << ' ';
		}
		std::cout << std::endl;
	}

	void InvalidateCache()
	{
		for (auto &entry : m_evalHash)
		{
			entry.hash = 0;
		}
	}

private:

	NNMatrixRM BoardsToFeatureRepresentation_(const std::vector<std::string> &positions, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions)
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

	template <typename Derived>
	NNMatrixRM ComputeSampleWeights_(const Eigen::MatrixBase<Derived> &x)
	{
		return m_mixingNet.ForwardPropagateFast(x);
	}

	NNMatrixRM ComputeExpertErrorDerivatives_(
		const NNMatrixRM &combinedPredictions,
		const NNMatrixRM &targets,
		const NNMatrixRM &sampleWeights,
		int64_t expertNum,
		const NNMatrixRM &finalLayerActivations)
	{
		// (y_combined - combinedPredictions) * (-gi(x)) * dtanh(act)/dz
		int64_t numExamples = combinedPredictions.rows();

		NNMatrixRM ret(numExamples, 1);

		// this takes care of everything except the dtanh(act)/dz term, which we can't really vectorize
		ret = ((targets - combinedPredictions).array() * (-sampleWeights.col(expertNum)).array()).matrix();

		// derivative of tanh is 1-tanh^2(x)
		for (int64_t i = 0; i < numExamples; ++i)
		{
			float tanhx = tanh(finalLayerActivations(i, 0));
			ret(i, 0) *= 1.0f - tanhx * tanhx;
		}

		return ret;
	}

	NNMatrixRM ComputeMixingNetErrorDerivatives_(
		const NNMatrixRM &combinedPredictions,
		const std::vector<NNMatrixRM> &indPredictions,
		const NNMatrixRM &targets,
		const NNMatrixRM &sampleWeights)
	{
		// (y_combined - combinedPredictions) * (-1) * (yi * gi * (1 - gi) + sum_over_all_models_k!=i{ -yk * gk * gi })
		int64_t numExamples = combinedPredictions.rows();
		int64_t numExperts = m_anns.size();

		NNMatrixRM ret(numExamples, numExperts);

		for (int64_t example = 0; example < numExamples; ++example)
		{
			for (int64_t expert = 0; expert < numExperts; ++expert)
			{
				float yi = indPredictions[expert](example, 0);
				float gi = sampleWeights(example, expert);
				float term1 = yi * gi * (1.0f - gi);

				float term2 = 0.0f;

				for (int64_t k = 0; k < numExperts; ++k)
				{
					if (k != expert)
					{
						float yk = indPredictions[k](example, 0);
						float gk = sampleWeights(example, k);
						term2 += -yk * gk * gi;
					}
				}

				ret(example, expert) = (targets(example, 0) - combinedPredictions(example, 0)) * (-1.0f);
				ret(example, expert) *= term1 + term2;
			}
		}

		return ret;
	}

	std::vector<EvalNet> m_anns;

	MixingNet m_mixingNet;

	std::vector<float> m_convTmp;

	std::vector<EvalHashEntry> m_evalHash;
};

#endif // ANN_EVALUATOR_H
