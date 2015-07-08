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

	ANNEvaluator() : m_evalHash(EvalHashSize)
	{
	}

	ANNEvaluator(const EvalNet &ann) : m_ann(ann), m_evalHash(EvalHashSize)
	{
	}

	ANNEvaluator(const std::string &filename) : m_evalHash(EvalHashSize)
	{
		std::ifstream netfIn(filename);
		DeserializeNet(m_ann, netfIn);
	}

	void BuildANN(const std::string &featureFilename, int64_t inputDims)
	{
		m_ann = LearnAnn::BuildEvalNet(featureFilename, inputDims);
	}

	EvalNet& GetANN()
	{
		InvalidateCache_();
		return m_ann;
	}

	const EvalNet& GetANN() const
	{
		return m_ann;
	}

	void Train(const NNMatrixRM &x, const NNMatrixRM &y)
	{
		m_ann.TrainGDM(x, y, 0.0f);
	}

	void TrainLoop(const NNMatrixRM &x, const NNMatrixRM &y, int64_t epochs)
	{
		LearnAnn::TrainANN(x, y, m_ann, epochs);
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

		Score nnRet = m_ann.ForwardPropagateSingle(mappedVec) * EvalFullScale;

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

	EvalNet m_ann;

	std::vector<float> m_convTmp;

	std::vector<EvalHashEntry> m_evalHash;
};

#endif // ANN_EVALUATOR_H
