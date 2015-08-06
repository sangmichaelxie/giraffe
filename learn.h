#ifndef LEARN_H
#define LEARN_H

#include <string>
#include <fstream>

#include <iostream>

#include "Eigen/Core"

namespace Learn
{

const static int64_t NumIterations = 1000000;
const static float TDLambda = 0.7f; // this is discount due to credit assignment uncertainty
const static float AbsLambda = 0.995f; // this is discount to encourage progress, and account for the snowball effect
const static int64_t HalfMovesToMake = 12;
const static size_t PositionsFirstBatch = 1000000;
const static size_t PositionsPerBatch = 100;
const static float MaxError = 0.15f;
const static int64_t SearchNodeBudget = 256;
const static float LearningRate = 1.0f;
const static float LearningRateSGD = 1.0f;
const static int64_t EvaluatorSerializeInterval = 100;
const static int64_t IterationPrintInterval = 10;

void TDL(const std::string &positionsFilename);

}

#endif // LEARN_H
