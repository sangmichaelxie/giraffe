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
	ANNEvaluator(const ANN &ann) : m_ann(ann) {}
	ANNEvaluator(const std::string &filename)
	{
		std::ifstream netfIn(filename);
		m_ann = DeserializeNet(netfIn);
	}

	Score Evaluate(const Board &b, Score lowerBound, Score upperBound)
	{
		std::vector<float> conv = FeaturesConv::ConvertBoardToNN<float>(b);

		Eigen::Map<NNVector> mappedVec(&conv[0], 1, conv.size());

		Score nnRet = m_ann.ForwardPropagateSingle(mappedVec);

		return b.GetSideToMove() == WHITE ? nnRet : -nnRet;
	}

private:
	ANN m_ann;
};

#endif // ANN_EVALUATOR_H
