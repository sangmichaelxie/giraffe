/*
	Copyright (C) 2015 Matthew Lai

	Giraffe is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	Giraffe is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FEATURES_CONV_H
#define FEATURES_CONV_H

#include <vector>
#include <string>
#include <sstream>
#include <set>

#include "Eigen/Dense"

#include "ann.h"
#include "board.h"
#include "types.h"
#include "containers.h"
#include "move.h"
#include "consts.h"

namespace FeaturesConv
{

struct FeatureDescription
{
	enum FeatureType
	{
		FeatureType_global, // global features are things like side to move, and material counts, and piece lists
		FeatureType_pos // property of a square
	};

	FeatureType featureType;

	// fields for global and pos features
	int32_t group;

	// fields for pos features
	Square sq;

	std::string ToString() const
	{
		std::stringstream ret;

		switch (featureType)
		{
		case FeatureType_global:
			ret << "GLOBAL ";
			ret << group << ' ';
			break;
		case FeatureType_pos:
			ret << "POS_GN ";
			ret << sq;
			break;
		default:
			assert(false);
		}

		return ret.str();
	}
};

// convert to NN input format
// T can either be float (to get actual values) or
// FeatureDescription (to get feature descriptions)
template <typename T>
void ConvertBoardToNN(Board &board, std::vector<T> &ret);

// additional info for conversion
struct ConvertMovesInfo
{
	std::vector<Score> see;
	std::vector<Score> nmSee; // SEE of the source square
};

// convert a list of moves to NN input format
void ConvertMovesToNN(
	Board &board,
	ConvertMovesInfo &convInfo,
	MoveList &ml,
	NNMatrixRM &ret);

// because of the way we convert a move list at a time, it's not possible to do the same thing with
// ConvertBoardToNN (using templatized functions to perform feature description extraction)
// so we need a separate function
void GetMovesFeatureDescriptions(std::vector<FeaturesConv::FeatureDescription> &fds);

}

#endif // FEATURES_CONV_H
