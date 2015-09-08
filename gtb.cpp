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

#include "gtb.h"

#include <iostream>
#include <string>
#include <sstream>

#include <cassert>
#include <cstdlib>

namespace
{

TB_squares SquareToTBSquare(Square sq)
{
	// luckily GTB uses the same order as Giraffe, so we only have to cast!
	return static_cast<TB_squares>(sq);
}

TB_pieces PieceTypeToTB(PieceType pt)
{
	switch (pt)
	{
	case WK:
	case BK:
		return tb_KING;
	case WQ:
	case BQ:
		return tb_QUEEN;
	case WR:
	case BR:
		return tb_ROOK;
	case WB:
	case BB:
		return tb_BISHOP;
	case WN:
	case BN:
		return tb_KNIGHT;
	case WP:
	case BP:
		return tb_PAWN;
	default:
		assert(false);
	}
}

void FillPieceListsPT(
	const Board &b,
	PieceType pt,
	unsigned int *squareList,
	unsigned char *pieceList,
	size_t &idx)
{
	uint64_t bb = b.GetPieceTypeBitboard(pt);

	while (bb)
	{
		Square sq = Extract(bb);
		bb &= ~(1ULL >> sq);

		squareList[idx] = SquareToTBSquare(sq);
		pieceList[idx] = PieceTypeToTB(pt);
		++idx;
	}
}

// fill the lists for TB access, and set tooMany to true if we have too many pieces so probe will fail
void FillPieceLists(
	const Board &b,
	unsigned int squareListWhite[17],
	unsigned char piecesListWhite[17],
	unsigned int squareListBlack[17],
	unsigned char piecesListBlack[17],
	bool &tooMany)
{
	tooMany = false;

	// there has to be a king
	size_t numWhite = 0;
	size_t numBlack = 0;

	FillPieceListsPT(b, WP, squareListWhite, piecesListWhite, numWhite);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }
	FillPieceListsPT(b, BP, squareListBlack, piecesListBlack, numBlack);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }

	FillPieceListsPT(b, WN, squareListWhite, piecesListWhite, numWhite);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }
	FillPieceListsPT(b, BN, squareListBlack, piecesListBlack, numBlack);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }

	FillPieceListsPT(b, WB, squareListWhite, piecesListWhite, numWhite);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }
	FillPieceListsPT(b, BB, squareListBlack, piecesListBlack, numBlack);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }

	FillPieceListsPT(b, WR, squareListWhite, piecesListWhite, numWhite);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }
	FillPieceListsPT(b, BR, squareListBlack, piecesListBlack, numBlack);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }

	FillPieceListsPT(b, WQ, squareListWhite, piecesListWhite, numWhite);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }
	FillPieceListsPT(b, BQ, squareListBlack, piecesListBlack, numBlack);
	if ((numWhite + numBlack) > (GTB::MaxPieces - 2)) { tooMany = true; return; }

	FillPieceListsPT(b, WK, squareListWhite, piecesListWhite, numWhite);
	FillPieceListsPT(b, BK, squareListBlack, piecesListBlack, numBlack);

	squareListWhite[numWhite] = tb_NOSQUARE;
	squareListBlack[numBlack] = tb_NOSQUARE;
	piecesListWhite[numWhite] = tb_NOPIECE;
	piecesListBlack[numBlack] = tb_NOPIECE;
}

}

namespace GTB
{

static bool initialized = false;
static const char **paths;

std::string Init(std::string path)
{
	if (path == "")
	{
		const char *envRet = getenv("GTBPath");

		if (envRet != nullptr)
		{
			path = envRet;
		}
	}

	if (path == "")
	{
		std::cout << "# GTBPath not set" << std::endl;
		return std::string();
	}

	paths = tbpaths_init();

	paths = tbpaths_add(paths, path.c_str());

	char *initInfo = tb_init(1, tb_CP4, paths);

	std::stringstream ssOut;

	if (initInfo != nullptr)
	{
		std::stringstream ss(initInfo);

		std::string line;
		while (std::getline(ss, line))
		{
			ssOut << "# " << line << std::endl;
		}
	}

	tbcache_init(CacheSize, WdlFraction);

	tbstats_reset();

	initialized = true;

	return ssOut.str();
}

ProbeResult Probe(const Board &b)
{
	ProbeResult ret;

	if (!initialized)
	{
		return ret;
	}

	// first we check total number of pawns, to rule out the majority of positions
	// if we have more than MaxPieces-2 pawns, the position won't be in TB (2 for 2 kings)
	if ((b.GetPieceCount(WP) + b.GetPieceCount(BP)) > (MaxPieces - 2))
	{
		return ret;
	}

	uint32_t stm = (b.GetSideToMove() == WHITE) ? tb_WHITE_TO_MOVE : tb_BLACK_TO_MOVE;
	unsigned int eps = b.IsEpAvailable() ? SquareToTBSquare(b.GetEpSquare()) : tb_NOSQUARE;

	unsigned int castle = 0;
	if (b.HasCastlingRight(W_SHORT_CASTLE)) castle |= tb_WOO;
	if (b.HasCastlingRight(W_LONG_CASTLE)) castle |= tb_WOOO;
	if (b.HasCastlingRight(B_SHORT_CASTLE)) castle |= tb_BOO;
	if (b.HasCastlingRight(B_LONG_CASTLE)) castle |= tb_BOOO;

	unsigned int squareListWhite[17];
	unsigned char piecesListWhite[17];
	unsigned int squareListBlack[17];
	unsigned char piecesListBlack[17];

	bool tooMany = false;
	FillPieceLists(b, squareListWhite, piecesListWhite, squareListBlack, piecesListBlack, tooMany);

	if (tooMany)
	{
		return ret;
	}

	unsigned int info = 0;
	unsigned int plies = 0;

	bool avail = false;

	#pragma omp critical(tbProbe)
	{
		avail = tb_probe_hard(
			stm,
			eps,
			castle,
			squareListWhite,
			squareListBlack,
			piecesListWhite,
			piecesListBlack,
			&info,
			&plies);
	}

	if (!avail)
	{
		return ret;
	}
	else if (info == tb_DRAW)
	{
		ret = 0;
	}
	else if ((info == tb_WMATE && b.GetSideToMove() == WHITE) || (info == tb_BMATE && b.GetSideToMove() == BLACK))
	{
		ret = MakeWinningScore(plies);
	}
	else if ((info == tb_WMATE && b.GetSideToMove() == BLACK) || (info == tb_BMATE && b.GetSideToMove() == WHITE))
	{
		ret = MakeLosingScore(plies);
	}
	else
	{
		std::cout << b.GetFen() << std::endl;
		assert(false);
	}

	return ret;
}

void DeInit()
{
	if (!initialized)
	{
		return;
	}

	tbpaths_done(paths);
	tbcache_done();
	tb_done();
}

}
