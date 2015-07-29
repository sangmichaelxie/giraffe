#include "features_conv.h"

#include <functional>

#include "see.h"
#include "move.h"

#include "bit_ops.h"

namespace
{

using namespace FeaturesConv;

typedef FixedVector<std::pair<Move, Score>, 31> MoveSEEList;

struct SEEValMap
{
	// maximum value of pieces white and black can put on each square
	int64_t whiteMaxVal[64];
	int64_t blackMaxVal[64];
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

#if 1
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
void PushThreat(std::vector<T> &ret, Square sq, Color c, bool exists, SEEValMap &seeMap, int32_t group)
{
	if (exists)
	{
		if (c == WHITE)
		{
			PushGlobalFloat(ret, NormalizeCount(seeMap.whiteMaxVal[sq], SEE::SEE_MAT[WK]), group);
		}
		else
		{
			PushGlobalFloat(ret, NormalizeCount(seeMap.blackMaxVal[sq], SEE::SEE_MAT[WK]), group);
		}
	}
	else
	{
		PushGlobalFloat(ret, 0.0f, group);
	}
}

template <typename T>
void PushAttacks(std::vector<T> &ret, Square sq, PieceType pt, bool exists, const Board &board, int32_t group, size_t numUsefulMoves[64])
{
	if (pt == WR || pt == BR || pt == WQ || pt == BQ)
	{
		// figure out how far we can go in each direction
		int32_t xStart = GetX(sq);
		int32_t yStart = GetY(sq);

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
		int32_t xStart = GetX(sq);
		int32_t yStart = GetY(sq);

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

	// 16 is the "reasonably maximum", though queens in the centre of an empty board can have up to 27. That's fine.
	PushMobility(ret, NormalizeCount(exists ? numUsefulMoves[sq] : 0, 16), group);
}

template <typename T>
void PushSquareFeatures(std::vector<T> &ret, const Board &board, SEEValMap &seeMap, int &group)
{
	// we store everything in arrays before actually pushing them, so that features
	// in the same group will be together (good for performance during eval)
	int64_t whiteMaterial[64] = { 0 };
	int64_t blackMaterial[64] = { 0 };

	for (Square sq = 0; sq < 64; ++sq)
	{
		PieceType pt = board.GetPieceAtSquare(sq);
		PieceType ptNc = StripColor(pt);
		Color c = GetColor(pt);

		if (pt != EMPTY)
		{
			switch (ptNc)
			{
			case WK:
				((c == WHITE) ? whiteMaterial[sq] : blackMaterial[sq]) = SEE::SEE_MAT[WK];
				break;
			case WQ:
				((c == WHITE) ? whiteMaterial[sq] : blackMaterial[sq]) = SEE::SEE_MAT[WQ];
				break;
			case WR:
				((c == WHITE) ? whiteMaterial[sq] : blackMaterial[sq]) = SEE::SEE_MAT[WR];
				break;
			case WB:
				((c == WHITE) ? whiteMaterial[sq] : blackMaterial[sq]) = SEE::SEE_MAT[WB];
				break;
			case WN:
				((c == WHITE) ? whiteMaterial[sq] : blackMaterial[sq]) = SEE::SEE_MAT[WN];
				break;
			case WP:
				((c	== WHITE) ? whiteMaterial[sq] : blackMaterial[sq]) = SEE::SEE_MAT[WP];
				break;
			}
		}
	}

	for (Square sq = 0; sq < 64; ++sq)
	{
		PushPosFloat(ret, sq, NormalizeCount(whiteMaterial[sq], SEE::SEE_MAT[WK]), group);
		PushPosFloat(ret, sq, NormalizeCount(blackMaterial[sq], SEE::SEE_MAT[WK]), group + 1);
		PushPosFloat(ret, sq, NormalizeCount(seeMap.blackMaxVal[sq], SEE::SEE_MAT[WK]), group + 2);
		PushPosFloat(ret, sq, NormalizeCount(seeMap.whiteMaxVal[sq], SEE::SEE_MAT[WK]), group + 3);
	}

	group += 4;
}

template <Color color, typename T>
void PushPawns(std::vector<T> &ret, uint64_t pawns, int32_t &group)
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

		//PushAttackIfValid(ret, sq, 1, (color == WHITE) ? 1 : -1, exists);
		//PushAttackIfValid(ret, sq, -1, (color == WHITE) ? 1 : -1, exists);

		PushGlobalCoords(ret, exists, sq, group);
		//++group;
	}
}

template <typename T>
void PushQueens(
	std::vector<T> &ret,
	uint64_t queens,
	PieceType pt,
	const Board &board,
	int32_t group,
	size_t numUsefulMoves[64],
	SEEValMap &seeMap,
	std::function<void(std::vector<T> &, int32_t)> pushFCFeaturesFcn)
{
	// queens (we only push the first queen for each side)
	if (queens)
	{
		uint32_t pos = BitScanForward(queens);

		PushGlobalCoords(ret, true, pos, group);
		PushAttacks(ret, pos, pt, true, board, group, numUsefulMoves);

		PushThreat(ret, pos, GetColor(pt), true, seeMap, group);
	}
	else
	{
		PushGlobalCoords(ret, false, 0, group);
		PushAttacks(ret, 0, pt, false, board, group, numUsefulMoves);

		PushThreat(ret, 0, WHITE, false, seeMap, group);
	}

	pushFCFeaturesFcn(ret, group);
}

template <typename T>
void PushPairPieces(
	std::vector<T> &ret,
	uint64_t pieces,
	PieceType pt,
	const Board &board,
	int32_t &group,
	size_t numUsefulMoves[64],
	SEEValMap &seeMap,
	std::function<void(std::vector<T> &, int32_t)> pushFCFeaturesFcn)
{
	// this is for rooks, bishops, and knights
	// for these pieces, we only look at the first 2, so there are
	// 3 possibilities - 0, 1, and 2

	size_t pieceCount = PopCount(pieces);

	if (pieceCount == 0)
	{
		PushGlobalCoords(ret, false, 0, group);
		PushAttacks(ret, 0, pt, false, board, group, numUsefulMoves);
		PushThreat(ret, 0, WHITE, false, seeMap, group);
		pushFCFeaturesFcn(ret, group);
		++group;
		PushGlobalCoords(ret, false, 0, group);
		PushAttacks(ret, 0, pt, false, board, group, numUsefulMoves);
		PushThreat(ret, 0, WHITE, false, seeMap, group);
		pushFCFeaturesFcn(ret, group);
	}
	else if (pieceCount == 1)
	{
		// we only have 1 piece
		// the slot we use depend on which half of the board it's on
		Square pos = Extract(pieces);
		int32_t x = GetX(pos);

		if (x < 4)
		{
			// use the first slot
			PushGlobalCoords(ret, true, pos, group);
			PushAttacks(ret, pos, pt, true, board, group, numUsefulMoves);
			PushThreat(ret, pos, GetColor(pt), true, seeMap, group);
			pushFCFeaturesFcn(ret, group);
			++group;
			PushGlobalCoords(ret, false, 0, group);
			PushAttacks(ret, pos, pt, false, board, group, numUsefulMoves);
			PushThreat(ret, 0, WHITE, false, seeMap, group);
			pushFCFeaturesFcn(ret, group);
		}
		else
		{
			// use the second slot
			PushGlobalCoords(ret, false, 0, group);
			PushAttacks(ret, pos, pt, false, board, group, numUsefulMoves);
			PushThreat(ret, 0, WHITE, false, seeMap, group);
			pushFCFeaturesFcn(ret, group);
			++group;
			PushGlobalCoords(ret, true, pos, group);
			PushAttacks(ret, pos, pt, true, board, group, numUsefulMoves);
			PushThreat(ret, pos, GetColor(pt), true, seeMap, group);
			pushFCFeaturesFcn(ret, group);
		}
	}
	else
	{
		// we have both pieces (or more)
		// the piece with lower x gets the first slot
		Square pos1 = Extract(pieces);
		Square pos2 = Extract(pieces);

		if (GetX(pos1) > GetX(pos2))
		{
			std::swap(pos1, pos2);
		}

		PushGlobalCoords(ret, true, pos1, group);
		PushAttacks(ret, pos1, pt, true, board, group, numUsefulMoves);
		PushThreat(ret, pos1, GetColor(pt), true, seeMap, group);
		pushFCFeaturesFcn(ret, group);
		++group;
		PushGlobalCoords(ret, true, pos2, group);
		PushAttacks(ret, pos2, pt, true, board, group, numUsefulMoves);
		PushThreat(ret, pos2, GetColor(pt), true, seeMap, group);
		pushFCFeaturesFcn(ret, group);
	}
}

SEEValMap ComputeSEEMaps(const Board &board)
{
	SEEValMap ret;

	Board boardCopy = board;

	for (Square sq = 0; sq < 64; ++sq)
	{
		if (board.GetSideToMove() == WHITE)
		{
			ret.blackMaxVal[sq] = SEE::SSEMap(boardCopy, sq);
			boardCopy.MakeNullMove();
			ret.whiteMaxVal[sq] = SEE::SSEMap(boardCopy, sq);
		}
		else
		{
			ret.whiteMaxVal[sq] = SEE::SSEMap(boardCopy, sq);
			boardCopy.MakeNullMove();
			ret.blackMaxVal[sq] = SEE::SSEMap(boardCopy, sq);
		}

		boardCopy.UndoMove();
	}

	return ret;
}

} // namespace

namespace FeaturesConv
{

template <typename T>
void ConvertBoardToNN(const Board &board, std::vector<T> &ret)
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

	SEEValMap seeMap = ComputeSEEMaps(board);

	// here we generate moves, and split them up by from square
	MoveList ml;
	board.GenerateAllMoves<Board::ALL>(ml);

	Board boardCopy = board;

	// each square has a list of moves originating from that square, with associated SEE score
	MoveSEEList mlByFromSq[64];
	size_t numUsefulMoves[64] = { 0 };

	for (size_t i = 0; i < ml.GetSize(); ++i)
	{
		Square fromSq = GetFromSquare(ml[i]);
		PieceType pt = GetPieceType(ml[i]);

		// don't worry about pawns for now, since we aren't using that information
		if (pt == WP || pt == BP)
		{
			continue;
		}

		Score see = SEE::StaticExchangeEvaluation(boardCopy, ml[i]);
		mlByFromSq[fromSq].PushBack(std::make_pair(ml[i], see));

		if (see >= 0)
		{
			++numUsefulMoves[fromSq];
		}
	}

	// now we can start actually forming the groups

	int32_t group = 0;

	// this function pushes important features that should go into all groups
	auto PushFCFeatures = [totalMat](std::vector<T> &ret, int32_t group)
	{
		PushGlobalFloat(ret, totalMat, group);
	};

	// first group contains piece counts and side to move
	// material (no need for king) -
	// these can also be calculated from "pieces exist" flags, they are
	// almost entirely redundant (except for set-up illegal positions, and promotions)
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

	++group;
	PushGlobalCoords(ret, true, wkPos, group, true);
	PushGlobalBool(ret, board.HasCastlingRight(W_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(W_LONG_CASTLE), group);
	PushFCFeatures(ret, group);

	++group;
	PushGlobalCoords(ret, true, bkPos, group, true);
	PushGlobalBool(ret, board.HasCastlingRight(B_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(B_LONG_CASTLE), group);
	PushFCFeatures(ret, group);

	// pawns (all pawns are in the same group)
	++group;
	PushPawns<WHITE>(ret, board.GetPieceTypeBitboard(WP), group);
	PushPawns<BLACK>(ret, board.GetPieceTypeBitboard(BP), group);
	PushFCFeatures(ret, group);

	// queens
	++group;
	PushQueens<T>(ret, board.GetPieceTypeBitboard(WQ), WQ, board, group, numUsefulMoves, seeMap, PushFCFeatures);
	++group;
	PushQueens<T>(ret, board.GetPieceTypeBitboard(BQ), BQ, board, group, numUsefulMoves, seeMap, PushFCFeatures);

	// rooks
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(WR), WR, board, group, numUsefulMoves, seeMap, PushFCFeatures);
	PushGlobalBool(ret, board.HasCastlingRight(W_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(W_LONG_CASTLE), group);
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(BR), BR, board, group, numUsefulMoves, seeMap, PushFCFeatures);
	PushGlobalBool(ret, board.HasCastlingRight(B_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(B_LONG_CASTLE), group);

	// bishops
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(WB), WB, board, group, numUsefulMoves, seeMap, PushFCFeatures);
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(BB), BB, board, group, numUsefulMoves, seeMap, PushFCFeatures);

	// knights
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(WN), WN, board, group, numUsefulMoves, seeMap, PushFCFeatures);
	++group;
	PushPairPieces<T>(ret, board.GetPieceTypeBitboard(BN), BN, board, group, numUsefulMoves, seeMap, PushFCFeatures);

	PushSquareFeatures(ret, board, seeMap, group);
}

template void ConvertBoardToNN<float>(const Board &board, std::vector<float> &ret);
template void ConvertBoardToNN<FeatureDescription>(const Board &board, std::vector<FeatureDescription> &ret);

} // namespace FeaturesConv
