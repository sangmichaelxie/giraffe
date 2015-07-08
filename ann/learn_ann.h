#ifndef LEARN_ANN_H
#define LEARN_ANN_H

#include "Eigen/Core"

#include "ann.h"

namespace LearnAnn
{

EvalNet BuildEvalNet(const std::string &featureFilename, int64_t inputDims);

template <typename Derived1, typename Derived2>
void TrainANN(
	const Eigen::MatrixBase<Derived1> &x,
	const Eigen::MatrixBase<Derived2> &y,
	EvalNet &nn,
	int64_t epochs);

void TrainANNFromFile(
	const std::string &xFilename,
	const std::string &yFilename,
	EvalNet &nn,
	int64_t epochs);

}

#endif // LEARN_ANN_H
