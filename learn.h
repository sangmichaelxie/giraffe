#ifndef LEARN_H
#define LEARN_H

#include <string>
#include <fstream>

#include <iostream>

#include "Eigen/Core"

namespace Learn
{

const static int64_t NumIterations = 10;

void TDL(const std::string &positionsFilename);

}

#endif // LEARN_H
