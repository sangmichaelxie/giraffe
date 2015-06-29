#ifndef ANN_EVALUATOR_H
#define ANN_EVALUATOR_H

#include <vector>
#include <string>

#include "evaluator.h"
#include "ann/ann.h"
#include "ann/features_conv.h"
#include "matrix_ops.h"

class ANNEvaluator : public EvaluatorIface
{
public:
	struct EvalHashEntry
	{
		uint64_t hash;
		Score val;
	};

	const static size_t EvalHashSize = 1*MB / sizeof(EvalHashEntry);

	ANNEvaluator()
	{
		m_evalHash.resize(EvalHashSize);
	}

	ANNEvaluator(const ANN &ann) : m_ann(ann)
	{
		m_evalHash.resize(EvalHashSize);
	}

	ANNEvaluator(const std::string &filename)
	{
		std::ifstream netfIn(filename);
		m_ann = DeserializeNet(netfIn);
		m_evalHash.resize(EvalHashSize);
	}

	Score EvaluateForWhite(const Board &b, Score lowerBound, Score upperBound)
	{
		uint64_t hash = b.GetHash();
		EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

		if (entry->hash == hash)
		{
			return entry->val;
		}

		FeaturesConv::ConvertBoardToNN(b, m_convTmp);

		Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

		Score nnRet = m_ann.ForwardPropagateSingle(mappedVec);

		entry->hash = hash;
		entry->val = nnRet;

		return nnRet;
	}

	ANN& GetANN()
	{
		return m_ann;
	}

private:
	ANN m_ann;

	std::vector<float> m_convTmp;

	std::vector<EvalHashEntry> m_evalHash;
};

#endif // ANN_EVALUATOR_H
