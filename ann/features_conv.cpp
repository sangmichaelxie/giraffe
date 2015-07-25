#include "features_conv.h"
#include "see.h"

#include "bit_ops.h"

namespace
{

using namespace FeaturesConv;

float NormalizeCoord(int x)
{
	// map x from 0 - 7 to -1 - 1
	return -1.0f + 0.2857f * x;
}

float NormalizeCount(int x, int typicalMaxCount)
{
	return -1.0f + (2.0f / typicalMaxCount) * x;
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
		ret.push_back(-1.0f);
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
void PushGlobalCoords(std::vector<T> &ret, bool exists, Square sq, int32_t group)
{
	PushGlobalBool(ret, exists, group);

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

template <typename T> void PushPosFloat(std::vector<T> &ret, Square pos, float x);
template<> void PushPosFloat<float>(std::vector<float> &ret, Square /*pos*/, float x)
{
	ret.push_back(x);
}

template<> void PushPosFloat<FeatureDescription>(std::vector<FeatureDescription> &ret, Square pos, float /*x*/)
{
	FeatureDescription fd;
	fd.featureType = FeatureDescription::FeatureType_pos;
	fd.sq = pos;
	ret.push_back(fd);
}

template <typename T>
void PushAttacks(std::vector<T> &ret, Square sq, PieceType pt, bool exists, const Board &board, int32_t group = 0)
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
}

template <typename T>
void PushSquareFeatures(std::vector<T> &ret, const Board &board)
{
	// we store everything in arrays before actually pushing them, so that features
	// in the same group will be together (good for performance during eval)
	int64_t whiteMaterial[64] = { 0 };
	int64_t blackMaterial[64] = { 0 };

	// smallest attacker of the square from both sides
	int64_t whiteSee[64] = { 0 };
	int64_t blackSee[64] = { 0 };

	Board boardCopy = board;

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

		if (board.GetSideToMove() == WHITE)
		{
			whiteSee[sq] = SEE::SSEMap(boardCopy, sq);
			boardCopy.MakeNullMove();
			blackSee[sq] = SEE::SSEMap(boardCopy, sq);
		}
		else
		{
			blackSee[sq] = SEE::SSEMap(boardCopy, sq);
			boardCopy.MakeNullMove();
			whiteSee[sq] = SEE::SSEMap(boardCopy, sq);
		}

		boardCopy.UndoMove();
	}

	for (Square sq = 0; sq < 64; ++sq)
	{
		PushPosFloat(ret, sq, NormalizeCount(whiteMaterial[sq], SEE::SEE_MAT[WK]));
		PushPosFloat(ret, sq, NormalizeCount(blackMaterial[sq], SEE::SEE_MAT[WK]));
		PushPosFloat(ret, sq, NormalizeCount(whiteSee[sq], SEE::SEE_MAT[WK]));
		PushPosFloat(ret, sq, NormalizeCount(blackSee[sq], SEE::SEE_MAT[WK]));
	}
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
void PushQueens(std::vector<T> &ret, uint64_t queens, PieceType pt, const Board &board, int32_t group)
{
	// queens (we only push the first queen for each side)
	if (queens)
	{
		uint32_t pos = BitScanForward(queens);

		PushGlobalCoords(ret, true, pos, group);
		PushAttacks(ret, pos, pt, true, board, group);
	}
	else
	{
		PushGlobalCoords(ret, false, 0, group);
		PushAttacks(ret, 0, pt, false, board, group);
	}
}

template <typename T>
void PushPairPieces(std::vector<T> &ret, uint64_t pieces, PieceType pt, const Board &board, int32_t &group)
{
	// this is for rooks, bishops, and knights
	// for these pieces, we only look at the first 2, so there are
	// 3 possibilities - 0, 1, and 2

	size_t pieceCount = PopCount(pieces);

	if (pieceCount == 0)
	{
		PushGlobalCoords(ret, false, 0, group);
		PushAttacks(ret, 0, pt, false, board, group);
		++group;
		PushGlobalCoords(ret, false, 0, group);
		PushAttacks(ret, 0, pt, false, board, group);
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
			PushAttacks(ret, pos, pt, true, board, group);
			++group;
			PushGlobalCoords(ret, false, 0, group);
			PushAttacks(ret, pos, pt, false, board, group);
		}
		else
		{
			// use the second slot
			PushGlobalCoords(ret, false, 0, group);
			PushAttacks(ret, pos, pt, false, board, group);
			++group;
			PushGlobalCoords(ret, true, pos, group);
			PushAttacks(ret, pos, pt, true, board, group);
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
		PushAttacks(ret, pos1, pt, true, board, group);
		++group;
		PushGlobalCoords(ret, true, pos2, group);
		PushAttacks(ret, pos2, pt, true, board, group);
	}
}

} // namespace

namespace FeaturesConv
{

template <typename T>
void ConvertBoardToNN(const Board &board, std::vector<T> &ret)
{
	ret.clear(); // this shouldn't actually deallocate memory

	int32_t group = 0;

	// material (no need for king) -
	// these can also be calculated from "pieces exist" flags, they are
	// almost entirely redundant (except for set-up illegal positions, and promotions)
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(WQ), 1.0f), group);
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(WR), 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(WB), 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(WN), 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(WP), 8.0f), group);
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(BQ), 1.0f), group);
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(BR), 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(BB), 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(BN), 2.0f), group);
	PushGlobalFloat(ret, NormalizeCount(board.GetPieceCount(BP), 8.0f), group);

	// which side to move
	PushGlobalBool(ret, board.GetSideToMove() == WHITE, group);

	// king positions
	uint32_t wkPos = board.GetFirstPiecePos(WK);
	uint32_t bkPos = board.GetFirstPiecePos(BK);

	++group;
	PushGlobalCoords(ret, true, wkPos, group);
	PushGlobalBool(ret, board.HasCastlingRight(W_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(W_LONG_CASTLE), group);

	++group;
	PushGlobalCoords(ret, true, bkPos, group);
	PushGlobalBool(ret, board.HasCastlingRight(B_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(B_LONG_CASTLE), group);

	// pawns
	++group;
	PushPawns<WHITE>(ret, board.GetPieceTypeBitboard(WP), group);
	++group;
	PushPawns<BLACK>(ret, board.GetPieceTypeBitboard(BP), group);

	// queens
	++group;
	PushQueens(ret, board.GetPieceTypeBitboard(WQ), WQ, board, group);
	++group;
	PushQueens(ret, board.GetPieceTypeBitboard(BQ), BQ, board, group);

	// rooks
	++group;
	PushPairPieces(ret, board.GetPieceTypeBitboard(WR), WR, board, group);
	PushGlobalBool(ret, board.HasCastlingRight(W_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(W_LONG_CASTLE), group);
	++group;
	PushPairPieces(ret, board.GetPieceTypeBitboard(BR), BR, board, group);
	PushGlobalBool(ret, board.HasCastlingRight(B_SHORT_CASTLE), group);
	PushGlobalBool(ret, board.HasCastlingRight(B_LONG_CASTLE), group);

	// bishops
	++group;
	PushPairPieces(ret, board.GetPieceTypeBitboard(WB), WB, board, group);
	++group;
	PushPairPieces(ret, board.GetPieceTypeBitboard(BB), BB, board, group);

	// knights
	++group;
	PushPairPieces(ret, board.GetPieceTypeBitboard(WN), WN, board, group);
	++group;
	PushPairPieces(ret, board.GetPieceTypeBitboard(BN), BN, board, group);

	PushSquareFeatures(ret, board);
}

template void ConvertBoardToNN<float>(const Board &board, std::vector<float> &ret);
template void ConvertBoardToNN<FeatureDescription>(const Board &board, std::vector<FeatureDescription> &ret);

} // namespace FeaturesConv
