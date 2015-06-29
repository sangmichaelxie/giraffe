#ifndef LEARN_ANN_H
#define LEARN_ANN_H

#include "Eigen/Core"

#include "ann_evaluator.h"

namespace LearnAnn
{

template <typename Derived>
ANN TrainANN(
	const Eigen::MatrixBase<Derived> &x,
	const Eigen::MatrixBase<Derived> &y,
	const std::string &featuresFilename);

// here we have to list all instantiations used (except for in learn_ann.cpp)
ANN TrainANN(const NNMatrixRM&, const NNVector&, const std::string);

ANN TrainANNFromFile(
	const std::string &xFilename,
	const std::string &yFilename,
	const std::string &featuresFilename);

}

#endif // LEARN_ANN_H
