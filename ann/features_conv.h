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
		FeatureType_global, // global features are things like side to move, and material counts
		FeatureType_posPieceType, // existence of a piece type at a square
		FeatureType_posMobility, // sliding mobility from a square
		FeatureType_pos // generic property of a square
	};

	FeatureType featureType;

	// fields for pos features
	Square sq;

	// fields for posPieceType
	PieceType pt;

	// fields for posMobility
	int32_t dirXOffset;
	int32_t dirYOffset;

	// fields for pos
	// (none)

	std::string ToString() const
	{
		std::stringstream ret;

		switch (featureType)
		{
		case FeatureType_global:
			ret << "GLOBAL";
			break;
		case FeatureType_posPieceType:
			ret << "POS_PT ";
			ret << sq << ' ';
			ret << PieceTypeToChar(pt);
			break;
		case FeatureType_posMobility:
			ret << "POS_MO ";
			ret << sq << ' ';
			ret << dirXOffset << ' ' << dirYOffset;
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
std::vector<T> ConvertBoardToNN(const Board &board);

std::set<Square> GetInfluences(const FeatureDescription &fd);

}

#endif // FEATURES_CONV_H
