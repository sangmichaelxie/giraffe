#ifndef LEARN_H
#define LEARN_H

#include <string>
#include <fstream>

#include <iostream>

#include "Eigen/Core"

namespace Learn
{

const static int64_t NumIterations = 1;
const static float Lambda = 0.7f;
const static int64_t HalfMovesToMake = 10;
const static size_t PositionsPerBatch = 20000;
const static float MaxError = 1000.0f;
const static int64_t SearchDepth = 1;
const static int64_t GamesPerIteration = 1; // each game produce about 150 positions
const static int64_t GamesFirstIteration = 1; // each game produce about 150 positions
const static size_t MaxHalfmovesPerGame = 200;

void TDL();

}

#endif // LEARN_H
