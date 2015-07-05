#ifndef LEARN_H
#define LEARN_H

#include <string>
#include <fstream>

#include <iostream>

#include "Eigen/Core"

namespace Learn
{

const static int64_t NumIterations = 30;
const static float Lambda = 0.85f;
const static int64_t HalfMovesToMake = 10;
const static size_t PositionsPerBatch = 250000;
const static float MaxError = 250.0f;

void TDL(const std::string &positionsFilename);

}

#endif // LEARN_H
