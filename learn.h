#ifndef LEARN_H
#define LEARN_H

#include <string>
#include <fstream>

#include <iostream>

#include "Eigen/Core"

namespace Learn
{

const static int64_t NumIterations = 20;
const static float Lambda = 0.7f;
const static int64_t FullMovesToMake = 2;
const static size_t MaxTrainingPositions = 300000;

void TDL(const std::string &positionsFilename);

}

#endif // LEARN_H
