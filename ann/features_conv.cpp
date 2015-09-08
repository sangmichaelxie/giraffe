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

#include "features_conv.h"

#include <functional>
#include <iomanip>

#include "see.h"
#include "move.h"

#include "bit_ops.h"

namespace
{

using namespace FeaturesConv;

typedef FixedVector<std::pair<Move, Score>, 31> MoveSEEList;

struct AttackMaps
{
	PieceType whiteLeastValuableAttackers[64];
	PieceType blackLeastValuableAttackers[64];

	uint8_t whiteNumAttackers[64];
	uint8_t blackNumAttackers[64];

	// normalized maximum value of pieces white and black can put on each square
	float whiteCtrl[64];
	float blackCtrl[64];

	// is it safe to move a piece of type pt to sq?
	bool IsSafe(PieceType pt, Square sq)
	{
		// here we are doing a very simple form of SEE
		// if the opponent has no attacker, the piece is safe
		// if the opponent has an attacker and it's lower valued, we are not safe
		// if the opponent has an attacker and it's equal or higher valued, we are
		// safe as long as we also have an attacker (that's not ourselves)

		// we don't have to worry about winning captures, because qsearch will take care of that
		// here we are only looking at moving to empty squares

		Color c = GetColor(pt);
		PieceType opponentAttacker = (c == WHITE) ? blackLeastValuableAttackers[sq] : whiteLeastValuableAttackers[sq];
		PieceType friendlyAttacker = (c == WHITE) ? whiteLeastValuableAttackers[sq] : blackLeastValuableAttackers[sq];
		uint8_t numFriendlyAttackers = (c == WHITE) ? whiteNumAttackers[sq] : blackNumAttackers[sq];

		if (opponentAttacker == EMPTY)
		{
			return true;
		}
		else if (SEE::SEE_MAT[opponentAttacker] < SEE::SEE_MAT[pt])
		{
			return false;
		}
		else
		{
			return friendlyAttacker != EMPTY && (numFriendlyAttackers > 1);
		}
	}
};

float NormalizeCoord(int x)
{
	// map x from 0 - 7 to 0 - 1
	return 0.1429 * x;
}

float NormalizeCount(int x, int typicalMaxCount)
{
	return static_cast<float>(x) / static_cast<float>(typicalMaxCount);
}

template <typename T> void PushGlobalBool(std::vector<T> &ret, bool x, int32_t group);
template<> void PushGlobalBool<float>(std::vector<float> &ret, bool x, int32_t /*group*/)
{
	if (x)
	{
		ret.push_back(1.0f);
	}
	else
	{
		ret.push_back(0.0f);
	}
}

template<> void PushGlobalBool<FeatureDescription>(std::vector<FeatureDescription> &ret, bool /*x*/, int32_t group)
{
	FeatureDescription fd;
	fd.featureType = FeatureDescription::FeatureType_global;
	fd.group = group;
	ret.push_back(fd);
}

template <typename T> void PushGlobalFloat(std::vector<T> &ret, float x, int32_t group);
template<> void PushGlobalFloat<float>(std::vector<float> &ret, float x, int32_t /*group*/)
{
	ret.push_back(x);
}

template<> void PushGlobalFloat<FeatureDescription>(std::vector<FeatureDescription> &ret, float /*x*/, int32_t group)
{
	FeatureDescription fd;
	fd.featureType = FeatureDescription::FeatureType_global;
	fd.group = group;
	ret.push_back(fd);
}

template <typename T>
void PushGlobalCoords(std::vector<T> &ret, bool exists, Square sq, int32_t group, bool mustExist = false)
{
	if (!mustExist)
	{
		PushGlobalBool(ret, exists, group);
	}

	uint32_t x = GetX(sq);
	uint32_t y = GetY(sq);

	PushGlobalFloat(ret, exists ? NormalizeCoord(x) : 0.0f, group);
	PushGlobalFloat(ret, exists ? NormalizeCoord(y) : 0.0f, group);

#if 0
	PushGlobalFloat(ret, exists ? NormalizeCount(GetDiag0(sq), 14) : 0.0f, group);
	PushGlobalFloat(ret, exists ? NormalizeCount(GetDiag1(sq), 14) : 0.0f, group);
#endif
}

template <typename T> void PushMobility(
		std::vector<T> &ret,
		float mob,
		int32_t group);
template<> void PushMobility<float>(
		std::vector<float> &ret,
		float mob,
		int32_t /*group*/)
{
	ret.push_back(mob);
}

template<> void PushMobility<FeatureDescription>(
		std::vector<FeatureDescription> &ret,
		float /*mob*/,
		int32_t group)
{
	FeatureDescription fd;

	fd.featureType = FeatureDescription::FeatureType_global;
	fd.group = group;

	ret.push_back(fd);
}

template <typename T> void PushPosFloat(std::vector<T> &ret, Square pos, float x, int group);
template<> void PushPosFloat<float>(std::vector<float> &ret, Square /*pos*/, float x, int /*group*/)
{
	ret.push_back(x);
}

template<> void PushPosFloat<FeatureDescription>(std::vector<FeatureDescription> &ret, Square pos, float /*x*/, int group)
{
	FeatureDescription fd;
	fd.featureType = FeatureDescription::FeatureType_pos;
	fd.sq = pos;
	fd.group = group;
	ret.push_back(fd);
}

template <typename T>
void PushAttacks(std::vector<T> &ret, Square sq, PieceType pt, bool exists, const Board &board, AttackMaps &atkMaps, int32_t group)
{
	int32_t safeMovesCount = 0;

	int32_t xStart = GetX(sq);
	int32_t yStart = GetY(sq);

	if (pt == WR || pt == BR || pt == WQ || pt == BQ)
	{
		// figure out how far we can go in each direction
		const int32_t DirXOffsets[4] = { 1, -1, 0, 0 };
		const int32_t DirYOffsets[4] = { 0, 0, 1, -1 };

		for (int32_t i = 0; i < 4; ++i)
		{
			// for each direction, keep going until we hit either the board edge or another piece
			int32_t count = 0;
			int32_t x = xStart + DirXOffsets[i];
			int32_t y = yStart + DirYOffsets[i];

			while (Valid(x) && Valid(y) && exists)
			{
				++count;

				if (atkMaps.IsSafe(pt, Sq(x, y)))
				{
					++safeMovesCount;
				}

				if (board.GetPieceAtSquare(Sq(x, y)) != EMPTY)
				{
					break;
				}

				x += DirXOffsets[i];
				y += DirYOffsets[i];
			}

			PushMobility(ret, NormalizeCount(count, 7), group);
		}
	}

	if (pt == WB || pt == BB || pt == WQ || pt == BQ)
	{
		// figure out how far we can go in each direction
		const int32_t DirXOffsets[4] = { 1, -1, 1, -1 };
		const int32_t DirYOffsets[4] = { 1, 1, -1, -1 };

		for (int32_t i = 0; i < 4; ++i)
		{
			// for each direction, keep going until we hit either the board edge or another piece
			int32_t count = 0;
			int32_t x = xStart + DirXOffsets[i];
			int32_t y = yStart + DirYOffsets[i];

			while (Valid(x) && Valid(y) && exists)
			{
				++count;

				if (atkMaps.IsSafe(pt, Sq(x, y)))
				{
					++safeMovesCount;
				}

				if (board.GetPieceAtSquare(Sq(x, y)) != EMPTY)
				{
					break;
				}

				x += DirXOffsets[i];
				y += DirYOffsets[i];
			}

			PushMobility(ret, NormalizeCount(count, 7), group);
		}
	}

	if (pt == WN || pt == BN)
	{
		const int32_t DirXOffsets[8] = { 2, -2, 1, -1, -2, 2, -1, 1 };
		const int32_t DirYOffsets[8] = { 1, 1, 2, 2, -1, -1, -2, -2 };

		for (int i = 0; i < 8; ++i)
		{
			int32_t x = xStart + DirXOffsets[i];
			int32_t y = yStart + DirYOffsets[i];

			if (Valid(x) && Valid(y) && exists)
			{
				if (atkMaps.IsSafe(pt, Sq(x, y)))
				{
					++safeMovesCount;
				}
			}
		}
	}

	// 16 is the "reasonably maximum", though queens in the centre of an empty board can have up to 27. That's fine.
	PushMobility(ret, NormalizeCount(exists ? safeMovesCount : 0, 16), group);
}

template <typename T>
void PushSquareFeatures(std::vector<T> &ret, const Board &/*board*/, AttackMaps &atkMaps, int &group)
{
	for (Square sq = 0; sq < 64; ++sq)
	{
		PushPosFloat(ret, sq, atkMaps.whiteCtrl[sq], group);
		PushPosFloat(ret, sq, atkMaps.blackCtrl[sq], group + 1);
	}

	group += 2;
}

template <Color color, typename T>
void PushPawns(std::vector<T> &ret, uint64_t pawns, AttackMaps &atkMaps, int32_t &group)
{
	std::tuple<bool, Square> assignments[8];

	for (size_t i = 0; i < 8; ++i)
	{
		std::get<0>(assignments[i]) = false;
	}

	// in the first pass, we assign each pawn to the corresponding file if possible,
	// and keep a list (in a bitboard) of pawns that still need to be assigned
	uint64_t unassigned = 0;

	while (pawns)
	{
		uint32_t thisPawn = Extract(pawns);

		uint32_t x = GetX(thisPawn);

		if (std::get<0>(assignments[x]) == false)
		{
			std::get<0>(assignments[x]) = true;
			std::get<1>(assignments[x]) = thisPawn;
		}
		else
		{
			unassigned |= 1LL << thisPawn;
		}
	}

	// then for each unassigned pawn (there should be very few),
	// look for the closest empty slot, and put it there
	while (unassigned)
	{
		uint32_t thisPawn = Extract(unassigned);

		uint32_t x = GetX(thisPawn);

		int32_t shortestDistance = 8;
		size_t bestSlot = 0;

		// find the nearest unoccupied slot
		for (int32_t i = 0; i < 8; ++i)
		{
			int32_t dist = abs(static_cast<int32_t>(x) - i);

			if (std::get<0>(assignments[i]) == false && dist < shortestDistance)
			{
				shortestDistance = dist;
				bestSlot = i;
			}
		}

		std::get<0>(assignments[bestSlot]) = true;
		std::get<1>(assignments[bestSlot]) = thisPawn;
	}

	for (size_t i = 0; i < 8; ++i)
	{
		bool exists = std::get<0>(assignments[i]);
		Square sq = std::get<1>(assignments[i]);

		PushGlobalCoords(ret, exists, sq, group);
		PushThreat(ret, sq, color, exists, atkMaps, group);
	}
}

template <typename T>
void PushThreat(
	std::vector<T> &ret,
	Square sq,
	Color /*c*/,
	bool exists,
	AttackMaps &atkMaps,
	int32_t group)
{
	if (exists)
	{
		// we push both black and white control because one would be defending the piece,
		// and one attacking
		PushGlobalFloat(ret, atkMaps.whiteCtrl[sq], group);
		PushGlobalFloat(ret, atkMaps.blackCtrl[sq], group);
	}
	else
	{
		PushGlobalFloat(ret, 0.0f, group);
		PushGlobalFloat(ret, 0.0f, group);
	}
}

template <typename T>
void PushQueens(
	std::vector<T> &ret,
	uint64_t queens,
	PieceType pt,
	const Board &board,
	int32_t group,
	std::function<void(std::vector<T> &, int32_t)> pushFCFeaturesFcn,
	AttackMaps &atkMaps)
{
	// queens (we only push the first queen for each side)
	bool exists = false;
	Square sq = 0;

	if (queens)
	{
		exists = true;
		sq = BitScanForward(queens);
	}

	PushGlobalCoords(ret, exists, sq, group);
	PushAttacks(ret, sq, pt, exists, board, atkMaps, group);
	PushThreat(ret, sq, GetColor(pt), exists, atkMaps, group);
	pushFCFeaturesFcn(ret, group);
}

template <typename T>
void PushPairPieces(
	std::vector<T> &ret,
	uint64_t pieces,
	PieceType pt,
	const Board &board,
	int32_t &group,
	std::function<void(std::vector<T> &, int32_t)> pushFCFeaturesFcn,
	AttackMaps &atkMaps)
{
	// this is for rooks, bishops, and knights
	// for these pieces, we only look at the first 2, so there are
	// 3 possibilities - 0, 1, and 2

	size_t pieceCount = PopCount(pieces);

	bool firstExists = false;
	bool secondExists = false;
	Square firstSq = 0;
	Square secondSq = 0;

	if (pieceCount == 0)
	{
		firstExists = false;
		secondExists = false;
	}
	else if (pieceCount == 1)
	{
		Square pos = Extract(pieces);
		int32_t x = GetX(pos);

		if (x < 4)
		{
			firstExists = true;
			firstSq = pos;
		}
		else
		{
			secondExists = true;
			secondSq = pos;
		}
	}
	else
	{
		firstSq = Extract(pieces);
		secondSq = Extract(pieces);

		firstExists = true;
		secondExists = true;

		if (GetX(firstSq) > GetX(secondSq))
		{
			std::swap(firstSq, secondSq);
		}
	}

	PushGlobalCoords(ret, firstExists, firstSq, group);
	PushAttacks(ret, firstSq, pt, firstExists, board, atkMaps, group);
	PushThreat(ret, firstSq, GetColor(pt), firstExists, atkMaps, group);
	pushFCFeaturesFcn(ret, group);
	++group;
	PushGlobalCoords(ret, secondExists, secondSq, group);
	PushAttacks(ret, secondSq, pt, secondExists, board, atkMaps, group);
	PushThreat(ret, secondSq, GetColor(pt), secondExists, atkMaps, group);
	pushFCFeaturesFcn(ret, group);
}

AttackMaps ComputeAttackMaps(Board &board)
{
	AttackMaps ret;

	board.ComputeLeastValuableAttackers(ret.whiteLeastValuableAttackers, ret.whiteNumAttackers, WHITE);
	board.ComputeLeastValuableAttackers(ret.blackLeastValuableAttackers, ret.blackNumAttackers, BLACK);

	// convert them to control values
	for (Square sq = 0; sq < 64; ++sq)
	{
		PieceType whitePt = ret.whiteLeastValuableAttackers[sq];
		PieceType blackPt = ret.blackLeastValuableAttackers[sq];

		// if a side doesn't attack the square, control is 0
		// if a side attacks with a piece, control is higher the lower valued the piece is
		ret.whiteCtrl[sq] = (whitePt == EMPTY) ? 0.0f : NormalizeCount(SEE::SEE_MAT[WK] + SEE::SEE_MAT[WK] / 2 - SEE::SEE_MAT[whitePt], SEE::SEE_MAT[WK] * 2);
		ret.blackCtrl[sq] = (blackPt == EMPTY) ? 0.0f : NormalizeCount(SEE::SEE_MAT[WK] + SEE::SEE_MAT[WK] / 2 - SEE::SEE_MAT[blackPt], SEE::SEE_MAT[WK] * 2);
	}

	return ret;
}

// push number of elements above, below, and equal to x
void PushRelativePlace(std::vector<float> &ret, const std::vector<float> &v, float x)
{
	float above = 0.0f;
	float below = 0.0f;
	float equal = 0.0f;

	for (const auto &v_x : v)
	{
		if (v_x > x)
		{
			above += 1.0f;
		}
		else if (v_x < x)
		{
			below += 1.0f;
		}
		else
		{
			equal += 1.0f;
		}
	}

	ret.push_back(above / static_cast<float>(v.size()));
	ret.push_back(below / static_cast<float>(v.size()));
	ret.push_back(equal / static_cast<float>(v.size()));
}

} // namespace

namespace FeaturesConv
{

template <typename T>
void ConvertBoardToNN(Board &board, std::vector<T> &ret)
{
	ret.clear(); // this shouldn't actually deallocate memory

	// we start by computing values that will be used later
	size_t WQCount = board.GetPieceCount(WQ);
	size_t WRCount = board.GetPieceCount(WR);
	size_t WBCount = board.GetPieceCount(WB);
	size_t WNCount = board.GetPieceCount(WN);
	size_t WPCount = board.GetPieceCount(WP);

	size_t BQCount = board.GetPieceCount(BQ);
	size_t BRCount = board.GetPieceCount(BR);
	size_t BBCount = board.GetPieceCount(BB);
	size_t BNCount = board.GetPieceCount(BN);
	size_t BPCount = board.GetPieceCount(BP);

	float WMatNP =
		SEE::SEE_MAT[WQ] * WQCount +
		SEE::SEE_MAT[WR] * WRCount +
		SEE::SEE_MAT[WB] * WBCount +
		SEE::SEE_MAT[WN] * WNCount;

	float BMatNP =
		SEE::SEE_MAT[BQ] * BQCount +
		SEE::SEE_MAT[BR] * BRCount +
		SEE::SEE_MAT[BB] * BBCount +
		SEE::SEE_MAT[BN] * BNCount;

	float WMatP = SEE::SEE_MAT[WP] * WPCount;
	float BMatP = SEE::SEE_MAT[BP] * BPCount;

	int64_t MaxTotalMatNP = SEE::SEE_MAT[WQ] + 2 * SEE::SEE_MAT[WR] + 2 * SEE::SEE_MAT[WB] + 2 * SEE::SEE_MAT[WN];
	int64_t MaxTotalMatP = 8 * SEE::SEE_MAT[WP];

	WMatNP = NormalizeCount(static_cast<int64_t>(WMatNP), MaxTotalMatNP);
	BMatNP = NormalizeCount(static_cast<int64_t>(BMatNP), MaxTotalMatNP);
	WMatP = NormalizeCount(static_cast<int64_t>(WMatP), MaxTotalMatP);
	BMatP = NormalizeCount(static_cast<int64_t>(BMatP), MaxTotalMatP);

	float totalMat = NormalizeCount(static_cast<int64_t>(WMatNP + BMatNP + WMatP + BMatP), MaxTotalMatNP * 2 + MaxTotalMatP * 2);

	// this function pushes important features that should go into all groups (currently empty)
	auto PushFCFeatures = [totalMat](std::vector<T> &/*ret*/, int32_t /*group*/)
	{
		//PushGlobalFloat(ret, totalMat, group);
	};

	AttackMaps atkMaps = ComputeAttackMaps(board);

	// now we can start actually forming the groups
	int32_t group = 0;

	// first group contains piece counts, side to move, and king positions
	// material (no need for king) -
	// these can also be calculated from "pieces exist" flags, they are
	// almost entirely redundant (except for set-up illegal positions, and promotions)
	// in evaluator, they are passed directly to second layer, because they are very important (game phase information)
	PushGlobalFloat(ret, NormalizeCount(WQCount, 1.0f), group);
	PushGlobalFloat(ret, NormalizeCount(WRCount, 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(WBCount, 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(WNCount, 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(WPCount, 8.0f), group);
	PushGlobalFloat(ret, NormalizeCount(BQCount, 1.0f), group);
	PushGlobalFloat(ret, NormalizeCount(BRCount, 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(BBCount, 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(BNCount, 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(BPCount, 8.0f), group);

	// which side to move
	PushGlobalBool(ret, board.GetSideToMove() == WHITE, group);

	// king positions
	uint32_t wkPos = board.GetFirstPiecePos(WK);
	uint32_t bkPos = board.GetFirstPiecePos(BK);

	PushGlobalCoords(ret, true, wkPos, group, true);
	PushGlobalBool(ret, board.HasCastlingRight(W_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(W_LONG_CASTLE), group);

	PushGlobalCoords(ret, true, bkPos, group, true);
	PushGlobalBool(ret, board.HasCastlingRight(B_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(B_LONG_CASTLE), group);

	// pawns (all pawns are in the same group)
	++group;
	PushPawns<WHITE>(ret, board.GetPieceTypeBitboard(WP), atkMaps, group);
	PushPawns<BLACK>(ret, board.GetPieceTypeBitboard(BP), atkMaps, group);
	PushFCFeatures(ret, group);

	// queens
	++group;
	PushQueens<T>(ret, board.GetPieceTypeBitboard(WQ), WQ, board, group, PushFCFeatures, atkMaps);
	++group;
	PushQueens<T>(ret, board.GetPieceTypeBitboard(BQ), BQ, board, group, PushFCFeatures, atkMaps);

	// rooks
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(WR), WR, board, group, PushFCFeatures, atkMaps);
	PushGlobalBool(ret, board.HasCastlingRight(W_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(W_LONG_CASTLE), group);
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(BR), BR, board, group, PushFCFeatures, atkMaps);
	PushGlobalBool(ret, board.HasCastlingRight(B_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(B_LONG_CASTLE), group);

	// bishops
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(WB), WB, board, group, PushFCFeatures, atkMaps);
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(BB), BB, board, group, PushFCFeatures, atkMaps);

	// knights
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(WN), WN, board, group, PushFCFeatures, atkMaps);
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(BN), BN, board, group, PushFCFeatures, atkMaps);

	PushSquareFeatures(ret, board, atkMaps, group);
}

template void ConvertBoardToNN<float>(Board &board, std::vector<float> &ret);
template void ConvertBoardToNN<FeatureDescription>(Board &board, std::vector<FeatureDescription> &ret);

void ConvertMovesToNN(Board &board, ConvertMovesInfo &convInfo, MoveList &ml, NNMatrixRM &ret)
{
	// first we generate the eval features to be shared between all moves
	// these features have to go to the end for performance, because all our new features will be group 0
	std::vector<float> sharedFeaturesBoard;
	ConvertBoardToNN(board, sharedFeaturesBoard);

	// shared features not specific to the board
	std::vector<float> sharedFeaturesOthers;

	// number of legal moves
	sharedFeaturesOthers.push_back(NormalizeCount(ml.GetSize(), 40));

	sharedFeaturesOthers.push_back(board.InCheck() ? 1.0f : 0.0f);

	std::vector<float> moveFeatures;

	// don't crash if the caller doesn't set SEE values
	convInfo.see.resize(ml.GetSize(), 0);
	convInfo.nmSee.resize(ml.GetSize(), 0);

	Color stm = board.GetSideToMove();

	for (size_t moveNum = 0; moveNum < ml.GetSize(); ++moveNum)
	{
		moveFeatures.clear();

		Move mv = ml[moveNum];

		Square from = GetFromSquare(mv);
		Square to = GetToSquare(mv);

		moveFeatures.push_back(NormalizeCoord(GetX(from)));
		moveFeatures.push_back(NormalizeCoord(GetEqY(from, stm)));
		moveFeatures.push_back(NormalizeCoord(GetX(to)));
		moveFeatures.push_back(NormalizeCoord(GetEqY(to, stm)));

		moveFeatures.push_back(board.IsViolent(mv) ? 1.0f : 0.0f);

		moveFeatures.push_back(board.IsChecking(mv) ? 1.0f : 0.0f);

		moveFeatures.push_back(convInfo.see[moveNum] > 0 ? 1.0f : 0.0f);
		moveFeatures.push_back(convInfo.see[moveNum] < 0 ? 1.0f : 0.0f);

		// positive value means we should move this, otherwise opponent can win it
		moveFeatures.push_back(convInfo.nmSee[moveNum] > 0 ? 1.0f : 0.0f);

		PieceType pt = GetPieceType(mv);

		bool isPT[6] = { false };

		assert(pt != EMPTY);

		isPT[COMPRESS_PT_IDX[StripColor(pt)]] = true;

		for (size_t i = 0; i < 6; ++i)
		{
			moveFeatures.push_back(isPT[i] ? 1.0f : 0.0f);
		}

		moveFeatures.insert(moveFeatures.end(), sharedFeaturesOthers.begin(), sharedFeaturesOthers.end());
		moveFeatures.insert(moveFeatures.end(), sharedFeaturesBoard.begin(), sharedFeaturesBoard.end());

		if (static_cast<size_t>(ret.cols()) != moveFeatures.size() || static_cast<size_t>(ret.rows()) != ml.GetSize())
		{
			ret.resize(ml.GetSize(), moveFeatures.size());
		}

		for (size_t i = 0; i < moveFeatures.size(); ++i)
		{
			ret(moveNum, i) = moveFeatures[i];
		}
	}
}

void GetMovesFeatureDescriptions(std::vector<FeaturesConv::FeatureDescription> &fds)
{
	ConvertMovesInfo convInfo;
	Board b;

	MoveList ml;
	b.GenerateAllLegalMoves<Board::ALL>(ml);

	NNMatrixRM x;

	ConvertMovesToNN(b, convInfo, ml, x);

	// this is the subset of features from ConvertBoardToNN
	std::vector<FeaturesConv::FeatureDescription> boardDescriptions;
	ConvertBoardToNN(b, boardDescriptions);

	int64_t numTotalFeatures = x.cols();
	int64_t numExtraFeatures = numTotalFeatures - boardDescriptions.size();

	// first we add the extra features (they are all group 0 global)
	for (int64_t featureNum = 0; featureNum < numExtraFeatures; ++featureNum)
	{
		FeaturesConv::FeatureDescription fd;
		fd.featureType = FeatureDescription::FeatureType_global;
		fd.group = 0;

		fds.push_back(fd);
	}

	// now we add the features shared with ConvertBoardToNN
	fds.insert(fds.end(), boardDescriptions.begin(), boardDescriptions.end());
}

} // namespace FeaturesConv
