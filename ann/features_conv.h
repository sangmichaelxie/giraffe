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
			ret << "POS_GN";
			ret << sq;
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
void ConvertBoardToNN(const Board &board, std::vector<T> &ret);

}

#endif // FEATURES_CONV_H
