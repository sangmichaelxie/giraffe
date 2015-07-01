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

	ANNEvaluator() : m_scaling(1.0f)
	{
		m_evalHash.resize(EvalHashSize);
	}

	ANNEvaluator(const ANN &ann) : m_ann(ann), m_scaling(1.0f)
	{
		m_evalHash.resize(EvalHashSize);
	}

	ANNEvaluator(const std::string &filename) : m_scaling(1.0f)
	{
		std::ifstream netfIn(filename);
		m_ann = DeserializeNet(netfIn);
		m_evalHash.resize(EvalHashSize);
	}

	Score EvaluateForWhiteImpl(const Board &b, Score lowerBound, Score upperBound)
	{
		uint64_t hash = b.GetHash();
		EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

		if (entry->hash == hash)
		{
			return entry->val;
		}

		FeaturesConv::ConvertBoardToNN(b, m_convTmp);

		Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

		Score nnRet = m_ann.ForwardPropagateSingle(mappedVec) * m_scaling;

		entry->hash = hash;
		entry->val = nnRet;

		return nnRet;
	}

	ANN& GetANN()
	{
		InvalidateCache_();
		return m_ann;
	}

	const ANN& GetANN() const
	{
		return m_ann;
	}

	void Calibrate()
	{
		// set m_scaling so that eval on "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
		// (black missing a7 pawn from start position) evaluates to 100
		Board calibratePosition("rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

		float v = EvaluateForWhiteImpl(calibratePosition, SCORE_MIN, SCORE_MAX);

		m_scaling = 100.0f / v;

		std::cout << "# Scaling factor: " << m_scaling << std::endl;

		InvalidateCache_();
	}

private:

	void InvalidateCache_()
	{
		for (auto &entry : m_evalHash)
		{
			entry.hash = 0;
		}
	}

	ANN m_ann;

	float m_scaling;

	std::vector<float> m_convTmp;

	std::vector<EvalHashEntry> m_evalHash;
};

#endif // ANN_EVALUATOR_H
