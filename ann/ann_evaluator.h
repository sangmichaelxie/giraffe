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

	const static size_t NumNets = 2;

	ANNEvaluator() : m_anns(NumNets), m_evalHash(EvalHashSize) {}

	ANNEvaluator(const std::vector<EvalNet> &anns) : m_anns(anns), m_evalHash(EvalHashSize)	{}

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
	}

	void Serialize(std::ostream &os)
	{
		os << m_anns.size() << std::endl;

		for (auto &net : m_anns)
		{
			SerializeNet(net, os);
		}
	}

	void Deserialize(std::istream &is)
	{
		size_t n;

		is >> n;

		m_anns.resize(n);

		for (auto &net : m_anns)
		{
			DeserializeNet(net, is);
		}
	}

	void Train(const std::vector<std::string> &positions, const NNMatrixRM &y, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions)
	{
		auto x = BoardsToFeatureRepresentation_(positions, featureDescriptions);

		auto weights = BoardsToSampleWeights_(positions);

		for (size_t i = 0; i < m_anns.size(); ++i)
		{
			m_anns[i].TrainGDM(x, y, weights.col(i), 0.0f);
		}
	}

	void TrainLoop(const std::vector<std::string> &positions, const NNMatrixRM &y, int64_t epochs, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions)
	{
		auto x = BoardsToFeatureRepresentation_(positions, featureDescriptions);

		auto weights = BoardsToSampleWeights_(positions);

		for (size_t i = 0; i < m_anns.size(); ++i)
		{
			LearnAnn::TrainANN(x, y, weights.col(i), m_anns[i], epochs);
		}
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

		float phase = GetPhase_(b);

		float nn0 = m_anns[0].ForwardPropagateSingle(mappedVec);
		float nn1 = m_anns[1].ForwardPropagateSingle(mappedVec);

		Score nnRet = (nn0 * phase + nn1 * (1.0f - phase)) * EvalFullScale;

		entry->hash = hash;
		entry->val = nnRet;

		return nnRet;
	}

private:
	void InvalidateCache_()
	{
		for (auto &entry : m_evalHash)
		{
			entry.hash = 0;
		}
	}

	float GetPhase_(const Board &b)
	{
		uint32_t WQCount = PopCount(b.GetPieceTypeBitboard(WQ));
		uint32_t WRCount = PopCount(b.GetPieceTypeBitboard(WR));
		uint32_t WBCount = PopCount(b.GetPieceTypeBitboard(WB));
		uint32_t WNCount = PopCount(b.GetPieceTypeBitboard(WN));

		uint32_t BQCount = PopCount(b.GetPieceTypeBitboard(BQ));
		uint32_t BRCount = PopCount(b.GetPieceTypeBitboard(BR));
		uint32_t BBCount = PopCount(b.GetPieceTypeBitboard(BB));
		uint32_t BNCount = PopCount(b.GetPieceTypeBitboard(BN));

		float sum =
			(WQCount + BQCount) * 4.0f +
			(WRCount + BRCount) * 2.0f +
			(WBCount + BBCount + WNCount + BNCount) * 1.0f;

		const float maxPhase =
			(2) * 4.0f +
			(4) * 2.0f +
			(8) * 1.0f;

		return std::min(sum / maxPhase, 1.0f);
	}

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

	NNMatrixRM BoardsToSampleWeights_(const std::vector<std::string> &positions)
	{
		NNMatrixRM ret(static_cast<int64_t>(positions.size()), static_cast<int64_t>(m_anns.size()));

		for (size_t i = 0; i < positions.size(); ++i)
		{
			float phase = GetPhase_(Board(positions[i]));
			ret(i, 0) = phase;
			ret(i, 1) = 1.0f - phase;
		}

		return ret;
	}

	std::vector<EvalNet> m_anns;

	std::vector<float> m_convTmp;

	std::vector<EvalHashEntry> m_evalHash;
};

#endif // ANN_EVALUATOR_H
