#ifndef LEARN_ANN_H
#define LEARN_ANN_H

#include "Eigen/Core"

#include "ann_evaluator.h"

namespace LearnAnn
{

template <typename Derived1, typename Derived2>
EvalNet TrainANN(
	const Eigen::MatrixBase<Derived1> &x,
	const Eigen::MatrixBase<Derived2> &y,
	const std::string &featuresFilename,
	EvalNet *start,
	int64_t epochs);

EvalNet TrainANNFromFile(
	const std::string &xFilename,
	const std::string &yFilename,
	const std::string &featuresFilename,
	EvalNet *start,
	int64_t epochs);

}

#endif // LEARN_ANN_H
