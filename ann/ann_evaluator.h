#ifndef ANN_EVALUATOR_H
#define ANN_EVALUATOR_H

#include <vector>
#include <string>

#include <cmath>

#include "evaluator.h"
#include "ann/ann.h"
#include "ann/features_conv.h"
#include "matrix_ops.h"
#include "consts.h"

#include "learn_ann.h"

//#define EVAL_HASH_STATS
//#define LAZY_EVAL

class ANNEvaluator : public EvaluatorIface
{
public:
	struct EvalHashEntry
	{
		uint64_t hash;
		Score val;

		enum class EntryType
		{
			EXACT,
			LOWERBOUND,
			UPPERBOUND
		} entryType;
	};

	const static size_t EvalHashSize = 32*MB / sizeof(EvalHashEntry);

	constexpr static float BoundNetErrorAsymmetry = 25.0f;

	constexpr static float BoundNetTargetShift = 0.03f;

	constexpr static float BoundEvalShift = 0.03f;

	ANNEvaluator();

	ANNEvaluator(const std::string &filename);

	void BuildANN(int64_t inputDims);

	void Serialize(std::ostream &os);

	void Deserialize(std::istream &is);

	void Train(const std::vector<std::string> &positions, const NNMatrixRM &y, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions, float learningRate);

	void TrainLoop(const std::vector<std::string> &positions, const NNMatrixRM &y, int64_t epochs, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions);

	void TrainBounds(const std::vector<std::string> &positions, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions, float learningRate);

	Score EvaluateForWhiteImpl(Board &b, Score lowerBound, Score upperBound) override;

	// we override this function to provide faster implementation using matrix-matrix multiplications instead of matrix-vector
	void BatchEvaluateForWhiteImpl(std::vector<Board> &positions, std::vector<Score> &results, Score lowerBound, Score upperBound) override;

	void PrintDiag(Board &board) override;

	void InvalidateCache();

	bool CheckBounds(Board &board, float &windowSize);

private:
	NNMatrixRM BoardsToFeatureRepresentation_(const std::vector<std::string> &positions, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions);

	NNMatrixRM ComputeErrorDerivatives_(
		const NNMatrixRM &predictions,
		const NNMatrixRM &targets,
		const NNMatrixRM &finalLayerActivations,
		float positiveWeight,
		float negativeWeight);

	Optional<Score> HashProbe_(const Board &b, Score lowerBound, Score upperBound)
	{
#ifdef EVAL_HASH_STATS
		static int64_t queries = 0;
		static int64_t exactHits = 0;
		static int64_t ubHits = 0;
		static int64_t lbHits = 0;
#endif

		Optional<Score> ret;

		uint64_t hash = b.GetHash();
		EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

		if (entry->hash == hash)
		{
			if (entry->entryType == EvalHashEntry::EntryType::EXACT)
			{
				ret = entry->val;

#ifdef EVAL_HASH_STATS
				++exactHits;
#endif
			}
			else if (entry->entryType == EvalHashEntry::EntryType::UPPERBOUND && entry->val <= lowerBound)
			{
				ret = entry->val;

#ifdef EVAL_HASH_STATS
				++ubHits;
#endif
			}
			else if (entry->entryType == EvalHashEntry::EntryType::LOWERBOUND && entry->val >= upperBound)
			{
				ret = entry->val;

#ifdef EVAL_HASH_STATS
				++lbHits;
#endif
			}
		}

#ifdef EVAL_HASH_STATS
		++queries;

		if (queries == 100000)
		{
			std::cout << "Queries: " << queries << std::endl;
			std::cout << "Exact hits: " << exactHits << std::endl;
			std::cout << "UB hits: " << ubHits << std::endl;
			std::cout << "LB hits: " << lbHits << std::endl;
		}
#endif

		return ret;
	}

	void HashStore_(const Board &b, Score score, EvalHashEntry::EntryType entryType)
	{
		uint64_t hash = b.GetHash();

		EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

		entry->hash = hash;
		entry->val = score;
		entry->entryType = entryType;
	}

	EvalNet m_mainAnn;

	EvalNet m_ubAnn;

	EvalNet m_lbAnn;

	std::vector<float> m_convTmp;

	std::vector<EvalHashEntry> m_evalHash;
};

#endif // ANN_EVALUATOR_H
