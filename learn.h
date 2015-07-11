#ifndef LEARN_H
#define LEARN_H

#include <string>
#include <fstream>

#include <iostream>

#include "Eigen/Core"

namespace Learn
{

const static int64_t NumIterations = 1000;
const static float Lambda = 0.7f;
const static int64_t HalfMovesToMake = 10;
const static size_t PositionsFirstBatch = 1000000;
const static size_t PositionsPerBatch = 1000;
const static float MaxError = 1000.0f;
const static int64_t SearchDepth = 2;
const static int64_t GamesPerIteration = 100; // each game produce about 150 positions
const static int64_t GamesFirstIteration = 1000; // each game produce about 150 positions
const static size_t MaxHalfmovesPerGame = 200;
const static float LearningRate = 0.5f;

void TDL(const std::string &positionsFilename);

}

#endif // LEARN_H
