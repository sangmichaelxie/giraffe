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

#include "board.h"

#include <string>
#include <set>
#include <sstream>
#include <regex>
#include <tuple>
#include <iostream>

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "bit_ops.h"
#include "containers.h"
#include "magic_moves.h"
#include "util.h"
#include "zobrist.h"

namespace
{

Square StringToSquare(std::string st)
{
	if ((st[0] > 'h') || (st[0] < 'a') || st[1] < '1' || st[1] > '8')
	{
		std::cerr << "Square is invalid - " << st << std::endl;
		exit(1);
	}

	return (st[1] - '1') * 8 + (st[0] - 'a');
}

PieceType CharToPieceType(char c)
{
	switch (c)
	{
	case 'K':
		return WK;
	case 'Q':
		return WQ;
	case 'B':
		return WB;
	case 'N':
		return WN;
	case 'R':
		return WR;
	case 'P':
		return WP;

	case 'k':
		return BK;
	case 'q':
		return BQ;
	case 'b':
		return BB;
	case 'n':
		return BN;
	case 'r':
		return BR;
	case 'p':
		return BP;

	default:
		std::cerr << "PieceType is invalid - " << c << std::endl;
		exit(1);
	}

	return EMPTY;
}

}

Board::Board(const std::string &fen)
{
	for (uint32_t i = 0; i < BOARD_DESC_BB_SIZE; ++i)
	{
		m_boardDescBB[i] = 0;
	}

	for (uint32_t i = 0; i < BOARD_DESC_U8_SIZE; ++i)
	{
		m_boardDescU8[i] = 0;
	}

	for (Square sq = 0; sq < 64; ++sq)
	{
		RemovePiece(sq);
	}

	// the board desc is up to 64 squares + 7 rank separators + 1 NUL
	char boardDesc[72];
	char sideToMove;
	char castlingRights[5];
	char enPassantSq[3];
	uint32_t halfMoves;
	uint32_t fullMoves;

	int sscanfRet = sscanf(fen.c_str(), "%71s %c %4s %2s %u %u",
						   boardDesc, &sideToMove, castlingRights, enPassantSq, &halfMoves, &fullMoves);

	if (sscanfRet == 6)
	{
		// all fields are present
	}
	else if (sscanfRet == 4) // some FEN/EPD positions don't have the move counts
	{
		halfMoves = 0;
		fullMoves = 1;
	}
	else
	{
		std::cerr << "FEN is invalid - " << fen << std::endl;
		exit(1);
	}

	size_t boardDescLen = strlen(boardDesc);
	Square currentSquareFen = 0;

	for (size_t i = 0; i < boardDescLen; ++i)
	{
		if (currentSquareFen >= 64)
		{
			std::cerr << "FEN is too long - " << fen << std::endl;
			exit(1);
		}

		// FEN goes from rank 8 to 1, so we have to reverse rank... a bit of weird logic here
		Square currentSquare = (7 - (currentSquareFen / 8)) * 8 + currentSquareFen % 8;

		if (boardDesc[i] == '/')
		{
			continue;
		}
		else if (boardDesc[i] >= '1' && boardDesc[i] <= '8')
		{
			for (size_t r = 0; r < static_cast<size_t>((boardDesc[i] - '0')); ++r)
			{
				RemovePiece(currentSquare);
				++currentSquareFen;
				++currentSquare;
			}
		}
		else if (boardDesc[i] >= 'A' && boardDesc[i] <= 'z')
		{
			PlacePiece(currentSquare, CharToPieceType(boardDesc[i]));
			++currentSquareFen;
		}
		else
		{
			std::cerr << "FEN is invalid (invalid character encountered) - " << fen << std::endl;
			exit(1);
		}
	}

	if (currentSquareFen != 64)
	{
		std::cerr << "FEN is invalid (too short) - " << fen << std::endl;
		exit(1);
	}

	if (sideToMove != 'w' && sideToMove != 'b')
	{
		std::cerr << "FEN is invalid (invalid side to move) - " << fen << std::endl;
		exit(1);
	}

	m_boardDescU8[SIDE_TO_MOVE] = sideToMove == 'w' ? WHITE : BLACK;

	if (enPassantSq[0] != '-')
	{
		m_boardDescBB[EN_PASS_SQUARE] = Bit(StringToSquare(std::string(enPassantSq)));
	}

	std::string strCastlingRights(castlingRights);

	m_boardDescU8[W_SHORT_CASTLE] = strCastlingRights.find('K') != std::string::npos;
	m_boardDescU8[W_LONG_CASTLE] = strCastlingRights.find('Q') != std::string::npos;
	m_boardDescU8[B_SHORT_CASTLE] = strCastlingRights.find('k') != std::string::npos;
	m_boardDescU8[B_LONG_CASTLE] = strCastlingRights.find('q') != std::string::npos;

	m_boardDescU8[HALF_MOVES_CLOCK] = halfMoves;

	UpdateInCheck_();
	UpdateHashFull_();

#ifdef DEBUG
	CheckBoardConsistency();
#endif
}

void Board::RemovePiece(Square sq)
{
	m_boardDescBB[m_boardDescU8[sq]] &= InvBit(sq);

	m_boardDescU8[sq] = EMPTY;

	// faster to reset both than check
	m_boardDescBB[WHITE_OCCUPIED] &= InvBit(sq);
	m_boardDescBB[BLACK_OCCUPIED] &= InvBit(sq);
}

void Board::PlacePiece(Square sq, PieceType pt)
{
#ifdef DEBUG
	assert(pt != EMPTY);
	assert(m_boardDescU8[sq] == EMPTY);
#endif

	m_boardDescU8[sq] = pt;
	m_boardDescBB[pt] |= Bit(sq);

	if (GetColor(pt) == WHITE)
	{
		m_boardDescBB[WHITE_OCCUPIED] |= Bit(sq);
	}
	else
	{
		m_boardDescBB[BLACK_OCCUPIED] |= Bit(sq);
	}
}

template <Board::MOVE_TYPES MT>
void Board::GenerateAllLegalMoves(MoveList &moveList)
{
	MoveList pseudoLegalMoves;
	GenerateAllPseudoLegalMoves_<MT>(pseudoLegalMoves);

	CheckInfo ci = ComputeCheckInfo();

	for (size_t i = 0; i < pseudoLegalMoves.GetSize(); ++i)
	{
		if (CheckLegal(ci, pseudoLegalMoves[i]))
		{
			moveList.PushBack(pseudoLegalMoves[i]);
		}
	}
}

template void Board::GenerateAllLegalMoves<Board::ALL>(MoveList &);
template void Board::GenerateAllLegalMoves<Board::VIOLENT>(MoveList &);
template void Board::GenerateAllLegalMoves<Board::QUIET>(MoveList &);

void Board::CheckBoardConsistency()
{
	for (uint32_t sq = 0; sq < 64; ++sq)
	{
		PieceType pt = m_boardDescU8[sq];
		if (pt == EMPTY)
		{
			for (uint32_t i = 0; i < NUM_PIECETYPES; ++i)
			{
				assert(!(m_boardDescBB[PIECE_TYPE_INDICES[i]] & Bit(sq)));
			}

			assert(!(m_boardDescBB[WHITE_OCCUPIED] & Bit(sq)));
			assert(!(m_boardDescBB[BLACK_OCCUPIED] & Bit(sq)));
		}
		else
		{
			if (GetColor(pt) == WHITE)
			{
				assert(m_boardDescBB[WHITE_OCCUPIED] & Bit(sq));
				assert(!(m_boardDescBB[BLACK_OCCUPIED] & Bit(sq)));
			}
			else
			{
				assert(m_boardDescBB[BLACK_OCCUPIED] & Bit(sq));
				assert(!(m_boardDescBB[WHITE_OCCUPIED] & Bit(sq)));
			}

			for (uint32_t i = 0; i < NUM_PIECETYPES; ++i)
			{
				if (PIECE_TYPE_INDICES[i] != pt)
				{
					assert(!(m_boardDescBB[PIECE_TYPE_INDICES[i]] & Bit(sq)));
				}
			}

			assert(m_boardDescBB[pt] & Bit(sq));
		}
	}

	if (m_boardDescU8[E1] != WK || m_boardDescU8[H1] != WR)
	{
		assert(!m_boardDescU8[W_SHORT_CASTLE]);
	}

	if (m_boardDescU8[E1] != WK || m_boardDescU8[A1] != WR)
	{
		assert(!m_boardDescU8[W_LONG_CASTLE]);
	}

	if (m_boardDescU8[E8] != BK || m_boardDescU8[H8] != BR)
	{
		if (m_boardDescU8[B_SHORT_CASTLE])
		{
			std::cout << PrintBoard() << std::endl;
		}
		assert(!m_boardDescU8[B_SHORT_CASTLE]);
	}

	if (m_boardDescU8[E8] != BK || m_boardDescU8[A8] != BR)
	{
		assert(!m_boardDescU8[B_LONG_CASTLE]);
	}

	uint64_t oldHash = GetHash();
	UpdateHashFull_();

	if (oldHash != GetHash())
	{
		std::cout << GetFen() << std::endl;
	}
	assert(oldHash == GetHash());
}

std::string Board::GetFen(bool omitMoveNums) const
{
	std::stringstream ss;

	for (int y = 7; y >= 0; --y)
	{
		for (int x = 0; x < 8;)
		{
			if (m_boardDescU8[Sq(x, y)] != EMPTY)
			{
				ss << PieceTypeToChar(m_boardDescU8[Sq(x, y)]);
				++x;
			}
			else
			{
				int numOfSpaces = 0;
				while (m_boardDescU8[Sq(x, y)] == EMPTY && x < 8)
				{
					++numOfSpaces;
					++x;
				}
				ss << numOfSpaces;
			}
		}

		if (y != 0)
		{
			ss << "/";
		}
	}

	ss << " ";

	ss << (m_boardDescU8[SIDE_TO_MOVE] == WHITE ? 'w' : 'b');

	ss << " ";

	if (!m_boardDescU8[W_SHORT_CASTLE] && !m_boardDescU8[W_LONG_CASTLE] && !m_boardDescU8[B_SHORT_CASTLE] && !m_boardDescU8[B_LONG_CASTLE])
	{
		ss << "-";
	}
	else
	{
		ss << (m_boardDescU8[W_SHORT_CASTLE] ? "K" : "") << (m_boardDescU8[W_LONG_CASTLE] ? "Q" : "")
			<< (m_boardDescU8[B_SHORT_CASTLE] ? "k" : "") << (m_boardDescU8[B_LONG_CASTLE] ? "q" : "");
	}

	ss << " ";

	if (m_boardDescBB[EN_PASS_SQUARE])
	{
		ss << SquareToString(BitScanForward(m_boardDescBB[EN_PASS_SQUARE]));
	}
	else
	{
		ss << "-";
	}

	if (!omitMoveNums)
	{
		ss << " " << m_boardDescU8[HALF_MOVES_CLOCK] << " " << 1; // we don't actually keep track of full moves
	}

	return ss.str();
}

std::string Board::PrintBoard() const
{
	std::stringstream ss;

	for (int y = 7; y >= 0; --y)
	{
		ss << "   ---------------------------------" << std::endl;
		ss << " " << (y + 1) << " |";

		for (int x = 0; x <= 7; ++x)
		{
				ss << " " << PieceTypeToChar(m_boardDescU8[Sq(x, y)]) << " |";
		}

		ss << std::endl;
	}

	ss << "   ---------------------------------" << std::endl;
	ss << "     a   b   c   d   e   f   g   h" << std::endl;

	ss << std::endl;

	ss << "Side to move: " << (m_boardDescU8[SIDE_TO_MOVE] == WHITE ? "white" : "black") << std::endl;
	ss << "En passant: " << (m_boardDescBB[EN_PASS_SQUARE] ? SquareToString(BitScanForward(m_boardDescBB[EN_PASS_SQUARE])) : "-") << std::endl;
	ss << "White castling rights: " << (m_boardDescU8[W_SHORT_CASTLE] ? "O-O " : "") << (m_boardDescU8[W_LONG_CASTLE] ? "O-O-O" : "") << std::endl;
	ss << "Black castling rights: " << (m_boardDescU8[B_SHORT_CASTLE] ? "O-O " : "") << (m_boardDescU8[B_LONG_CASTLE] ? "O-O-O" : "") << std::endl;

	ss << "Half moves since last pawn move or capture: " << m_boardDescU8[HALF_MOVES_CLOCK] << std::endl;
	ss << "FEN: " << GetFen() << std::endl;
	ss << "In check: " << m_boardDescU8[IN_CHECK] << std::endl;
	ss << "Insufficient material: " << HasInsufficientMaterial() << std::endl;

	return ss.str();
}

bool Board::ApplyMove(Move mv)
{
#define MOVE_PIECE(pt, from, to) \
	m_boardDescBB[pt] ^= Bit(from) | Bit(to); \
	m_boardDescU8[from] = EMPTY; \
	m_boardDescU8[to] = pt;
#define REMOVE_PIECE(pt, sq) \
	m_boardDescBB[pt] &= InvBit(sq); \
	m_boardDescU8[sq] = EMPTY;
#define PLACE_PIECE(pt, sq) \
	m_boardDescBB[pt] |= Bit(sq); \
	m_boardDescU8[sq] = pt;
#define REPLACE_PIECE(pt_old, pt_new, sq) \
	m_boardDescBB[pt_old] &= InvBit(sq); \
	m_boardDescBB[pt_new] |= Bit(sq); \
	m_boardDescU8[sq] = pt_new;

	UndoListBB &ulBB = m_undoStackBB.PrePush();
	UndoListU8 &ulU8 = m_undoStackU8.PrePush();

	ulBB.Clear();
	ulU8.Clear();

	PieceType pt = GetPieceType(mv);
	Square from = GetFromSquare(mv);
	Square to = GetToSquare(mv);
	Color color = pt & COLOR_MASK;
	PieceType promoType = GetPromoType(mv);

	m_hashStack.Push(m_boardDescBB[HASH]);

	if (m_boardDescBB[EN_PASS_SQUARE])
	{
		m_boardDescBB[HASH] ^= EN_PASS_ZOBRIST[BitScanForward(m_boardDescBB[EN_PASS_SQUARE])];
	}

	ulBB.PushBack(std::make_pair(EN_PASS_SQUARE, m_boardDescBB[EN_PASS_SQUARE]));
	uint64_t currentEp = m_boardDescBB[EN_PASS_SQUARE];
	m_boardDescBB[EN_PASS_SQUARE] = 0;

	ulU8.PushBack(std::make_pair(IN_CHECK, m_boardDescU8[IN_CHECK]));

	if (IsCastling(mv))
	{
		if (GetCastlingType(mv) == MoveConstants::CASTLE_WHITE_SHORT)
		{
			ulU8.PushBack(std::make_pair(E1, m_boardDescU8[E1]));
			ulU8.PushBack(std::make_pair(G1, m_boardDescU8[G1]));
			ulU8.PushBack(std::make_pair(H1, m_boardDescU8[H1]));
			ulU8.PushBack(std::make_pair(F1, m_boardDescU8[F1]));
			ulBB.PushBack(std::make_pair(WK, m_boardDescBB[WK]));
			ulBB.PushBack(std::make_pair(WR, m_boardDescBB[WR]));
			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED, m_boardDescBB[WHITE_OCCUPIED]));

			if (m_boardDescU8[W_SHORT_CASTLE])
			{
				m_boardDescBB[HASH] ^= W_SHORT_CASTLE_ZOBRIST;
			}

			if (m_boardDescU8[W_LONG_CASTLE])
			{
				m_boardDescBB[HASH] ^= W_LONG_CASTLE_ZOBRIST;
			}

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[E1][WK];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[G1][WK];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[H1][WR];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[F1][WR];

			MOVE_PIECE(WK, E1, G1);
			MOVE_PIECE(WR, H1, F1);
			ulU8.PushBack(std::make_pair(W_SHORT_CASTLE, m_boardDescU8[W_SHORT_CASTLE]));
			ulU8.PushBack(std::make_pair(W_LONG_CASTLE, m_boardDescU8[W_LONG_CASTLE]));
			m_boardDescU8[W_SHORT_CASTLE] = 0;
			m_boardDescU8[W_LONG_CASTLE] = 0;
			m_boardDescBB[WHITE_OCCUPIED] ^= Bit(E1) | Bit(G1) | Bit(H1) | Bit(F1);
		}
		else if (GetCastlingType(mv) == MoveConstants::CASTLE_WHITE_LONG)
		{
			ulU8.PushBack(std::make_pair(E1, m_boardDescU8[E1]));
			ulU8.PushBack(std::make_pair(C1, m_boardDescU8[C1]));
			ulU8.PushBack(std::make_pair(A1, m_boardDescU8[A1]));
			ulU8.PushBack(std::make_pair(D1, m_boardDescU8[D1]));
			ulBB.PushBack(std::make_pair(WK, m_boardDescBB[WK]));
			ulBB.PushBack(std::make_pair(WR, m_boardDescBB[WR]));
			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED, m_boardDescBB[WHITE_OCCUPIED]));

			if (m_boardDescU8[W_SHORT_CASTLE])
			{
				m_boardDescBB[HASH] ^= W_SHORT_CASTLE_ZOBRIST;
			}

			if (m_boardDescU8[W_LONG_CASTLE])
			{
				m_boardDescBB[HASH] ^= W_LONG_CASTLE_ZOBRIST;
			}

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[E1][WK];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[C1][WK];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[A1][WR];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[D1][WR];

			MOVE_PIECE(WK, E1, C1);
			MOVE_PIECE(WR, A1, D1);
			ulU8.PushBack(std::make_pair(W_SHORT_CASTLE, m_boardDescU8[W_SHORT_CASTLE]));
			ulU8.PushBack(std::make_pair(W_LONG_CASTLE, m_boardDescU8[W_LONG_CASTLE]));
			m_boardDescU8[W_SHORT_CASTLE] = 0;
			m_boardDescU8[W_LONG_CASTLE] = 0;
			m_boardDescBB[WHITE_OCCUPIED] ^= Bit(E1) | Bit(C1) | Bit(A1) | Bit(D1);
		}
		else if (GetCastlingType(mv) == MoveConstants::CASTLE_BLACK_SHORT)
		{
			ulU8.PushBack(std::make_pair(E8, m_boardDescU8[E8]));
			ulU8.PushBack(std::make_pair(G8, m_boardDescU8[G8]));
			ulU8.PushBack(std::make_pair(H8, m_boardDescU8[H8]));
			ulU8.PushBack(std::make_pair(F8, m_boardDescU8[F8]));
			ulBB.PushBack(std::make_pair(BK, m_boardDescBB[BK]));
			ulBB.PushBack(std::make_pair(BR, m_boardDescBB[BR]));
			ulBB.PushBack(std::make_pair(BLACK_OCCUPIED, m_boardDescBB[BLACK_OCCUPIED]));

			if (m_boardDescU8[B_SHORT_CASTLE])
			{
				m_boardDescBB[HASH] ^= B_SHORT_CASTLE_ZOBRIST;
			}

			if (m_boardDescU8[B_LONG_CASTLE])
			{
				m_boardDescBB[HASH] ^= B_LONG_CASTLE_ZOBRIST;
			}

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[E8][BK];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[G8][BK];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[H8][BR];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[F8][BR];

			MOVE_PIECE(BK, E8, G8);
			MOVE_PIECE(BR, H8, F8);
			ulU8.PushBack(std::make_pair(B_SHORT_CASTLE, m_boardDescU8[B_SHORT_CASTLE]));
			ulU8.PushBack(std::make_pair(B_LONG_CASTLE, m_boardDescU8[B_LONG_CASTLE]));
			m_boardDescU8[B_SHORT_CASTLE] = 0;
			m_boardDescU8[B_LONG_CASTLE] = 0;
			m_boardDescBB[BLACK_OCCUPIED] ^= Bit(E8) | Bit(G8) | Bit(H8) | Bit(F8);
		}
		else // (GetCastlingType(mv) == MoveConstants::CASTLE_BLACK_LONG)
		{
			ulU8.PushBack(std::make_pair(E8, m_boardDescU8[E8]));
			ulU8.PushBack(std::make_pair(C8, m_boardDescU8[C8]));
			ulU8.PushBack(std::make_pair(A8, m_boardDescU8[A8]));
			ulU8.PushBack(std::make_pair(D8, m_boardDescU8[D8]));
			ulBB.PushBack(std::make_pair(BK, m_boardDescBB[BK]));
			ulBB.PushBack(std::make_pair(BR, m_boardDescBB[BR]));
			ulBB.PushBack(std::make_pair(BLACK_OCCUPIED, m_boardDescBB[BLACK_OCCUPIED]));

			if (m_boardDescU8[B_SHORT_CASTLE])
			{
				m_boardDescBB[HASH] ^= B_SHORT_CASTLE_ZOBRIST;
			}

			if (m_boardDescU8[B_LONG_CASTLE])
			{
				m_boardDescBB[HASH] ^= B_LONG_CASTLE_ZOBRIST;
			}

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[E8][BK];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[C8][BK];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[A8][BR];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[D8][BR];

			MOVE_PIECE(BK, E8, C8);
			MOVE_PIECE(BR, A8, D8);
			ulU8.PushBack(std::make_pair(B_SHORT_CASTLE, m_boardDescU8[B_SHORT_CASTLE]));
			ulU8.PushBack(std::make_pair(B_LONG_CASTLE, m_boardDescU8[B_LONG_CASTLE]));
			m_boardDescU8[B_SHORT_CASTLE] = 0;
			m_boardDescU8[B_LONG_CASTLE] = 0;
			m_boardDescBB[BLACK_OCCUPIED] ^= Bit(E8) | Bit(C8) | Bit(A8) | Bit(D8);
		}
	}
	else if ((pt == WP || pt == BP) && Bit(to) == currentEp) // en passant
	{
		if (pt == WP)
		{
			ulU8.PushBack(std::make_pair(from, m_boardDescU8[from]));
			ulU8.PushBack(std::make_pair(to, m_boardDescU8[to]));
			ulU8.PushBack(std::make_pair(to - 8, m_boardDescU8[to - 8]));
			ulBB.PushBack(std::make_pair(WP, m_boardDescBB[WP]));
			ulBB.PushBack(std::make_pair(BP, m_boardDescBB[BP]));

			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED, m_boardDescBB[WHITE_OCCUPIED]));
			ulBB.PushBack(std::make_pair(BLACK_OCCUPIED, m_boardDescBB[BLACK_OCCUPIED]));

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[from][WP];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to][WP];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to - 8][BP];

			MOVE_PIECE(WP, from, to);
			REMOVE_PIECE(BP, to - 8);
			m_boardDescBB[WHITE_OCCUPIED] ^= (Bit(from) | Bit(to));
			m_boardDescBB[BLACK_OCCUPIED] ^= Bit(to - 8);
		}
		else
		{
			ulU8.PushBack(std::make_pair(from, m_boardDescU8[from]));
			ulU8.PushBack(std::make_pair(to, m_boardDescU8[to]));
			ulU8.PushBack(std::make_pair(to + 8, m_boardDescU8[to + 8]));
			ulBB.PushBack(std::make_pair(WP, m_boardDescBB[WP]));
			ulBB.PushBack(std::make_pair(BP, m_boardDescBB[BP]));

			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED, m_boardDescBB[WHITE_OCCUPIED]));
			ulBB.PushBack(std::make_pair(BLACK_OCCUPIED, m_boardDescBB[BLACK_OCCUPIED]));

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[from][BP];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to][BP];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to + 8][WP];

			MOVE_PIECE(BP, from, to);
			REMOVE_PIECE(WP, to + 8);
			m_boardDescBB[BLACK_OCCUPIED] ^= (Bit(from) | Bit(to));
			m_boardDescBB[WHITE_OCCUPIED] ^= Bit(to + 8);
		}
	}
	else
	{
		int32_t dy = GetY(from) - GetY(to);

		bool isCapture = m_boardDescU8[to] != EMPTY; // this is only for NON-EP captures
		bool isPromotion = promoType != 0;
		bool isPawnDoubleMove = (pt == WP || pt == BP) && (dy != 1 && dy != -1);

		if (isCapture && !isPromotion)
		{
			ulU8.PushBack(std::make_pair(from, m_boardDescU8[from]));
			ulU8.PushBack(std::make_pair(to, m_boardDescU8[to]));
			ulBB.PushBack(std::make_pair(pt, m_boardDescBB[pt]));
			ulBB.PushBack(std::make_pair(m_boardDescU8[to], m_boardDescBB[m_boardDescU8[to]]));

			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED, m_boardDescBB[WHITE_OCCUPIED]));
			ulBB.PushBack(std::make_pair(BLACK_OCCUPIED, m_boardDescBB[BLACK_OCCUPIED]));

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[from][pt];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to][pt];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to][m_boardDescU8[to]];

			REMOVE_PIECE(pt, from);
			REPLACE_PIECE(m_boardDescU8[to], pt, to);

			m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)] ^= Bit(to);
			m_boardDescBB[WHITE_OCCUPIED | color] ^= Bit(to) | Bit(from);
		}
		else if (!isPromotion && !isCapture)
		{
			ulU8.PushBack(std::make_pair(from, m_boardDescU8[from]));
			ulU8.PushBack(std::make_pair(to, m_boardDescU8[to]));
			ulBB.PushBack(std::make_pair(pt, m_boardDescBB[pt]));

			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED | color, m_boardDescBB[WHITE_OCCUPIED | color]));

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[from][pt];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to][pt];

			MOVE_PIECE(pt, from, to);
			m_boardDescBB[WHITE_OCCUPIED | color] ^= Bit(to) | Bit(from);
		}
		else if (isPromotion && isCapture)
		{
			ulU8.PushBack(std::make_pair(from, m_boardDescU8[from]));
			ulU8.PushBack(std::make_pair(to, m_boardDescU8[to]));
			ulBB.PushBack(std::make_pair(pt, m_boardDescBB[pt]));
			ulBB.PushBack(std::make_pair(m_boardDescU8[to], m_boardDescBB[m_boardDescU8[to]]));
			ulBB.PushBack(std::make_pair(promoType, m_boardDescBB[promoType]));

			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED, m_boardDescBB[WHITE_OCCUPIED]));
			ulBB.PushBack(std::make_pair(BLACK_OCCUPIED, m_boardDescBB[BLACK_OCCUPIED]));

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[from][pt];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to][promoType];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to][m_boardDescU8[to]];

			REMOVE_PIECE(pt, from);
			REPLACE_PIECE(m_boardDescU8[to], promoType, to);

			m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)] ^= Bit(to);
			m_boardDescBB[WHITE_OCCUPIED | color] ^= Bit(to) | Bit(from);
		}
		else // !isCapture && isPromotion
		{
			ulU8.PushBack(std::make_pair(from, m_boardDescU8[from]));
			ulU8.PushBack(std::make_pair(to, m_boardDescU8[to]));
			ulBB.PushBack(std::make_pair(pt, m_boardDescBB[pt]));
			ulBB.PushBack(std::make_pair(promoType, m_boardDescBB[promoType]));

			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED | color, m_boardDescBB[WHITE_OCCUPIED | color]));

			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[from][pt];
			m_boardDescBB[HASH] ^= PIECES_ZOBRIST[to][promoType];

			REMOVE_PIECE(pt, from);
			PLACE_PIECE(promoType, to);
			m_boardDescBB[WHITE_OCCUPIED | color] ^= Bit(to) | Bit(from);
		}

		// check for pawn move (update ep)
		if (isPawnDoubleMove)
		{
			// this was saved to undo list earlier already
			m_boardDescBB[EN_PASS_SQUARE] = PAWN_MOVE_1[from][pt == WP ? 0 : 1];
			m_boardDescBB[HASH] ^= EN_PASS_ZOBRIST[BitScanForward(PAWN_MOVE_1[from][pt == WP ? 0 : 1])];
		}

		// update castling rights
		if (m_boardDescU8[W_SHORT_CASTLE] && (pt == WK || (pt == WR && from == H1) || (to == H1)))
		{
			ulU8.PushBack(std::make_pair(W_SHORT_CASTLE, m_boardDescU8[W_SHORT_CASTLE]));
			m_boardDescU8[W_SHORT_CASTLE] = 0;

			m_boardDescBB[HASH] ^= W_SHORT_CASTLE_ZOBRIST;
		}

		if (m_boardDescU8[W_LONG_CASTLE] && (pt == WK || (pt == WR && from == A1) || (to == A1)))
		{
			ulU8.PushBack(std::make_pair(W_LONG_CASTLE, m_boardDescU8[W_LONG_CASTLE]));
			m_boardDescU8[W_LONG_CASTLE] = 0;

			m_boardDescBB[HASH] ^= W_LONG_CASTLE_ZOBRIST;
		}

		if (m_boardDescU8[B_SHORT_CASTLE] && (pt == BK || (pt == BR && from == H8) || (to == H8)))
		{
			ulU8.PushBack(std::make_pair(B_SHORT_CASTLE, m_boardDescU8[B_SHORT_CASTLE]));
			m_boardDescU8[B_SHORT_CASTLE] = 0;

			m_boardDescBB[HASH] ^= B_SHORT_CASTLE_ZOBRIST;
		}

		if (m_boardDescU8[B_LONG_CASTLE] && (pt == BK || (pt == BR && from == A8) || (to == A8)))
		{
			ulU8.PushBack(std::make_pair(B_LONG_CASTLE, m_boardDescU8[B_LONG_CASTLE]));
			m_boardDescU8[B_LONG_CASTLE] = 0;

			m_boardDescBB[HASH] ^= B_LONG_CASTLE_ZOBRIST;
		}

		// update half move clock
		// castling does not reset the clock
		ulU8.PushBack(std::make_pair(HALF_MOVES_CLOCK, m_boardDescU8[HALF_MOVES_CLOCK]));
		if (isCapture || pt == WP || pt == BP)
		{
			m_boardDescU8[HALF_MOVES_CLOCK] = 0;
		}
		else
		{
			if (m_boardDescU8[HALF_MOVES_CLOCK] != std::numeric_limits<uint8_t>::max())
			{
				++m_boardDescU8[HALF_MOVES_CLOCK];
			}
		}
	}

#ifdef DEBUG
	CheckBoardConsistency();

	// verify that we don't have duplicate entries in the undo lists
	std::set<uint32_t> entries;
	for (size_t i = 0; i < ulBB.GetSize(); ++i)
	{
		if (entries.find(ulBB[i].first) != entries.end())
		{
			std::cout << "Duplicate undo entry in BB undo list found! - " << static_cast<int>(ulBB[i].first) << std::endl;
			assert(false);
		}

		entries.insert(ulBB[i].first);
	}

	entries.clear();
	for (size_t i = 0; i < ulU8.GetSize(); ++i)
	{
		if (entries.find(ulU8[i].first) != entries.end())
		{
			std::cout << "Duplicate undo entry in U8 undo list found! - " << ulU8[i].first << std::endl;
			assert(false);
		}

		entries.insert(ulU8[i].first);
	}

	// verify that we have updated the hash correctly
	uint64_t oldHash = GetHash();
	UpdateHashFull_();

	if (oldHash != GetHash())
	{
		std::cout << GetFen() << std::endl;
		std::cout << MoveToAlg(mv) << std::endl;
	}
	assert(oldHash == GetHash());
#endif

	UpdateInCheck_();

	if (InCheck())
	{
		// this position is illegal, undo the move
		for (size_t i = 0; i < ulBB.GetSize(); ++i)
		{
			m_boardDescBB[ulBB[i].first] = ulBB[i].second;
		}

		// this position is illegal, undo the move
		for (size_t i = 0; i < ulU8.GetSize(); ++i)
		{
			m_boardDescU8[ulU8[i].first] = ulU8[i].second;
		}

		m_boardDescBB[HASH] = m_hashStack.Pop();

		m_undoStackBB.Pop();
		m_undoStackU8.Pop();

		return false;
	}

	// no need to store this
	m_boardDescU8[SIDE_TO_MOVE] = m_boardDescU8[SIDE_TO_MOVE] ^ COLOR_MASK;
	m_boardDescBB[HASH] ^= SIDE_TO_MOVE_ZOBRIST;

	UpdateInCheck_(); // this is for the new side

	m_moveStack.Push(mv);

	return true;

#undef MOVE_PIECE
#undef REMOVE_PIECE
}

Board::CheckInfo Board::ComputeCheckInfo()
{
	CheckInfo ret;

	Square kingPos = (GetSideToMove() == WHITE ? BitScanForward(m_boardDescBB[WK]) : BitScanForward(m_boardDescBB[BK]));
	int32_t kingX = GetX(kingPos);
	int32_t kingY = GetY(kingPos);
	int32_t kingDiag0 = GetDiag0(kingPos);
	int32_t kingDiag1 = GetDiag1(kingPos);

	uint64_t opponentQueens = (GetSideToMove() == WHITE ? m_boardDescBB[BQ] : m_boardDescBB[WQ]);
	uint64_t opponentRooks = (GetSideToMove() == WHITE ? m_boardDescBB[BR] : m_boardDescBB[WR]);
	uint64_t opponentBishops = (GetSideToMove() == WHITE ? m_boardDescBB[BB] : m_boardDescBB[WB]);

	while (opponentQueens)
	{
		Square pos = Extract(opponentQueens);

		// all the cases are mutually exclusive
		if (GetX(pos) == kingX)
		{
			ret.opponentRQOnSameX = true;
		}
		else if (GetY(pos) == kingY)
		{
			ret.opponentRQOnSameY = true;
		}
		else if (GetDiag0(pos) == kingDiag0)
		{
			ret.opponentBQOnSameDiag0 = true;
		}
		else if (GetDiag1(pos) == kingDiag1)
		{
			ret.opponentBQOnSameDiag1 = true;
		}
	}

	while (opponentRooks)
	{
		Square pos = Extract(opponentRooks);

		// all the cases are mutually exclusive
		if (GetX(pos) == kingX)
		{
			ret.opponentRQOnSameX = true;
		}
		else if (GetY(pos) == kingY)
		{
			ret.opponentRQOnSameY = true;
		}
	}

	while (opponentBishops)
	{
		Square pos = Extract(opponentBishops);

		// all the cases are mutually exclusive
		if (GetDiag0(pos) == kingDiag0)
		{
			ret.opponentBQOnSameDiag0 = true;
		}
		else if (GetDiag1(pos) == kingDiag1)
		{
			ret.opponentBQOnSameDiag1 = true;
		}
	}

	return ret;
}

bool Board::CheckLegal(const CheckInfo &ci, Move mv)
{
	// in this function we try to determine legality of moves quickly without doing a full check (by applying the move)
	PieceType pt = GetPieceType(mv);

	bool doFullCheck = false;

	if (InCheck())
	{
		// if we are in check, do a full check because there are many potential ways to get out of check, and we are lazy
		doFullCheck = true;
	}
	else
	{
		switch (pt)
		{
		case WK:
		case BK:
			// if this is a king move, we do a full check because any king move can potentially put the king in check
			// this includes castling
			doFullCheck = true;
			break;
		case WP:
		case BP:
			if (Bit(GetToSquare(mv)) == m_boardDescBB[EN_PASS_SQUARE])
			{
				// if this is EP, we do a full check... because EPs are rare, and we are lazy
				doFullCheck = true;
				break;
			}
			// fall through if this is a pawn move that's not EP
		default:
			{
				doFullCheck = false;

				// now we know we are not moving the king, this is not EP, and the king is not already in check
				// in this case, the only way this move can put the king in check is if we just moved away a pinned piece
				// besides EP and castling (both have been dealt with), the only square that will be vacated after a move is the from square

				Square vacatedSquare = GetFromSquare(mv);
				Square kingPos = BitScanForward(GetSideToMove() == WHITE ? m_boardDescBB[WK] : m_boardDescBB[BK]);

				if ((ci.opponentRQOnSameX && (GetX(vacatedSquare) == GetX(kingPos))) ||
					(ci.opponentRQOnSameY && (GetY(vacatedSquare) == GetY(kingPos))) ||
					(ci.opponentBQOnSameDiag0 && (GetDiag0(vacatedSquare) == GetDiag0(kingPos))) ||
					(ci.opponentBQOnSameDiag1 && (GetDiag1(vacatedSquare) == GetDiag1(kingPos))))
				{
					doFullCheck = true;
				}
			}
		}
	}

	if (doFullCheck)
	{
		bool ret = ApplyMove(mv);
		if (ret)
		{
			UndoMove();
			return true;
		}
		else
		{
			return false;
		}
	}

	return true;
}

void Board::UndoMove()
{
	UndoListBB &ulBB = m_undoStackBB.Top();
	UndoListU8 &ulU8 = m_undoStackU8.Top();

	// this is the only thing not stored in the undo list
	m_boardDescU8[SIDE_TO_MOVE] = m_boardDescU8[SIDE_TO_MOVE] ^ COLOR_MASK;

	for (size_t i = 0; i < ulBB.GetSize(); ++i)
	{
		m_boardDescBB[ulBB[i].first] = ulBB[i].second;
	}

	for (size_t i = 0; i < ulU8.GetSize(); ++i)
	{
		m_boardDescU8[ulU8[i].first] = ulU8[i].second;
	}

	m_boardDescBB[HASH] = m_hashStack.Pop();

	m_undoStackBB.Pop();
	m_undoStackU8.Pop();

	m_moveStack.Pop();
}

std::string Board::MoveToAlg(Move mv) const
{
	Square from = GetFromSquare(mv);
	Square to = GetToSquare(mv);
	PieceType promo = GetPromoType(mv);

	std::string ret;

	ret += SquareToString(from);
	ret += SquareToString(to);

	if (promo != 0)
	{
		ret += static_cast<char>(tolower(PieceTypeToChar(promo)));
	}

	return ret;
}

bool Board::operator==(const Board &other)
{
	for (size_t i = 0; i < BOARD_DESC_BB_SIZE; ++i)
	{
		if (m_boardDescBB[i] != other.m_boardDescBB[i])
		{
			//std::cout << i << std::endl;
			//DebugPrint(m_boardDesc[i]);
			//DebugPrint(other.m_boardDesc[i]);
			return false;
		}
	}

	for (size_t i = 0; i < BOARD_DESC_U8_SIZE; ++i)
	{
		if (m_boardDescU8[i] != other.m_boardDescU8[i])
		{
			//std::cout << i << std::endl;
			//DebugPrint(m_boardDesc[i]);
			//DebugPrint(other.m_boardDesc[i]);
			return false;
		}
	}

	return true;
}

Move Board::ParseMove(std::string str)
{
	MoveList moveList;
	GenerateAllLegalMoves<ALL>(moveList);

	if (PatternMatch(str, "[a-h][1-8][a-h][1-8]"))
	{
		char srcX;
		int srcY;
		char dstX;
		int dstY;

		if (sscanf(str.c_str(), "%c%d%c%d", &srcX, &srcY, &dstX, &dstY) == 4)
		{
			srcX -= 'a';
			dstX -= 'a';
			srcY -= 1;
			dstY -= 1;

			for (size_t i = 0; i < moveList.GetSize(); ++i)
			{
				Square from = GetFromSquare(moveList[i]);
				Square to = GetToSquare(moveList[i]);

				if (from == Sq(srcX, srcY) && to == Sq(dstX, dstY))
				{
					return moveList[i];
				}
			}
		}
	}

	// algebraic promotion
	if (PatternMatch(str, "[a-h][1-8][a-h][1-8][QBRNqbrn]"))
	{
		char srcX;
		int srcY;
		char dstX;
		int dstY;
		char promo;

		if (sscanf(str.c_str(), "%c%d%c%d%c", &srcX, &srcY, &dstX, &dstY, &promo) == 5)
		{
			PieceType promoType = WQ;
			switch (promo)
			{
			case 'Q':
			case 'q':
				promoType = WQ;
				break;
			case 'B':
			case 'b':
				promoType = WB;
				break;
			case 'N':
			case 'n':
				promoType = WN;
				break;
			case 'R':
			case 'r':
				promoType = WR;
				break;
			}

			promoType = static_cast<PieceType>(promoType | m_boardDescU8[SIDE_TO_MOVE]);

			srcX -= 'a';
			dstX -= 'a';
			srcY -= 1;
			dstY -= 1;

			for (size_t i = 0; i < moveList.GetSize(); ++i)
			{
				Square from = GetFromSquare(moveList[i]);
				Square to = GetToSquare(moveList[i]);

				if (from == Sq(srcX, srcY) && to == Sq(dstX, dstY) && promoType == GetPromoType(moveList[i]))
				{
					return moveList[i];
				}
			}
		}
	}

	return 0;
}

bool Board::IsZugzwangProbable()
{
	if (m_boardDescU8[SIDE_TO_MOVE] == WHITE)
	{
		return !(
			m_boardDescBB[WR] ||
			m_boardDescBB[WQ] ||
			m_boardDescBB[WB] ||
			m_boardDescBB[WN]);
	}
	else
	{
		return !(
			m_boardDescBB[BR] ||
			m_boardDescBB[BQ] ||
			m_boardDescBB[BB] ||
			m_boardDescBB[BN]);
	}
}

void Board::MakeNullMove()
{
	assert(!InCheck());

	UndoListBB &undoListBB = m_undoStackBB.PrePush();
	UndoListU8 &undoListU8 = m_undoStackU8.PrePush(); // this is empty, but we still need to push

	undoListBB.Clear();
	undoListU8.Clear();

	m_hashStack.Push(m_boardDescBB[HASH]);

	m_moveStack.Push(0);

	// this doesn't need to be stored in the undo stack
	m_boardDescU8[SIDE_TO_MOVE] ^= COLOR_MASK;

	if (m_boardDescBB[EN_PASS_SQUARE])
	{
		undoListBB.PushBack(std::make_pair(EN_PASS_SQUARE, m_boardDescBB[EN_PASS_SQUARE]));
		m_boardDescBB[HASH] ^= EN_PASS_ZOBRIST[BitScanForward(m_boardDescBB[EN_PASS_SQUARE])];
		m_boardDescBB[EN_PASS_SQUARE] = 0;
	}

	undoListBB.PushBack(std::make_pair(HASH, m_boardDescBB[HASH]));
	m_boardDescBB[HASH] ^= SIDE_TO_MOVE_ZOBRIST;

	UpdateInCheck_();

#ifdef DEBUG
	CheckBoardConsistency();
#endif
}

bool Board::CheckPseudoLegal(Move mv)
{
	PieceType pt = GetPieceType(mv);
	PieceType ptNoColor = StripColor(pt);

	Square from = GetFromSquare(mv);
	Square to = GetToSquare(mv);
	Color color = pt & COLOR_MASK;

	PieceType toPt = m_boardDescU8[to];
	Color toColor = m_boardDescU8[to] & COLOR_MASK;

	uint64_t totalOccupancy = m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED];

	// there is no legal move where the destination is occupied by a friendly piece
	if (toPt != EMPTY && toColor == color)
	{
		return false;
	}

	// if the from piece doesn't exist...
	if (pt != m_boardDescU8[from])
	{
		return false;
	}

	// wrong side to move
	if (color != m_boardDescU8[SIDE_TO_MOVE])
	{
		return false;
	}

	if (ptNoColor == WN)
	{
		// for knights, the only requirement is the destination square must be enemy or empty
		// we checked that already
		return true;
	}
	else if (ptNoColor == WR)
	{
		// for rooks, we actually have to do move generation, because there may be an additional blocker
		return Rmagic(from, totalOccupancy) & Bit(to);
	}
	else if (ptNoColor == WB)
	{
		// for bishops, we actually have to do move generation, because there may be an additional blocker
		return Bmagic(from, totalOccupancy) & Bit(to);
	}
	else if (ptNoColor == WQ)
	{
		// for queens, we actually have to do move generation, because there may be an additional blocker
		return Qmagic(from, totalOccupancy) & Bit(to);
	}
	else if (ptNoColor == WP)
	{
		// if from and to are on different files, this must be a capture or en passant
		if (GetX(from) != GetX(to))
		{
			return (toPt != EMPTY) || (m_boardDescBB[EN_PASS_SQUARE] && to == BitScanForward(m_boardDescBB[EN_PASS_SQUARE])); // we have already checked for friendly
		}
		else if ((GetY(from) - GetY(to)) == 1 || (GetY(from) - GetY(to)) == -1)
		{
			// regular move
			return toPt == EMPTY;
		}
		else
		{
			// only other move type is a 2 square push, in which case both the destination and square jumped over
			// must be empty
			Square midSquare = (from + to) / 2;
			return toPt == EMPTY && m_boardDescU8[midSquare] == EMPTY;
		}
	}
	else
	{
		// for kings
		// castling is the only special case here
		if (from == E1 && to == G1)
		{
			return	m_boardDescU8[W_SHORT_CASTLE] &&
					m_boardDescU8[H1] == WR &&
					m_boardDescU8[F1] == EMPTY &&
					m_boardDescU8[G1] == EMPTY &&
					!IsUnderAttack_(E1) &&
					!IsUnderAttack_(F1);
		}
		else if (from == E1 && to == C1)
		{
			return	m_boardDescU8[W_LONG_CASTLE] &&
					m_boardDescU8[A1] == WR &&
					m_boardDescU8[B1] == EMPTY &&
					m_boardDescU8[C1] == EMPTY &&
					m_boardDescU8[D1] == EMPTY &&
					!IsUnderAttack_(E1) &&
					!IsUnderAttack_(D1);
		}
		else if (from == E8 && to == G8)
		{
			return	m_boardDescU8[B_SHORT_CASTLE] &&
					m_boardDescU8[H8] == BR &&
					m_boardDescU8[F8] == EMPTY &&
					m_boardDescU8[G8] == EMPTY &&
					!IsUnderAttack_(E8) &&
					!IsUnderAttack_(F8);
		}
		else if (from == E8 && to == C8)
		{
			return	m_boardDescU8[B_LONG_CASTLE] &&
					m_boardDescU8[A8] == BR &&
					m_boardDescU8[B8] == EMPTY &&
					m_boardDescU8[C8] == EMPTY &&
					m_boardDescU8[D8] == EMPTY &&
					!IsUnderAttack_(E8) &&
					!IsUnderAttack_(D8);
		}
		else
		{
			// for normal moves, destination needs to be enemy or empty, and we checked that already
			return true;
		}
	}
}

bool Board::IsViolent(Move mv)
{
	bool isQPromo = GetPromoType(mv) == WQ || GetPromoType(mv) == BQ;
	bool isCapture = m_boardDescU8[GetToSquare(mv)] != EMPTY || Bit(GetToSquare(mv)) == m_boardDescBB[EN_PASS_SQUARE];

	return isQPromo || isCapture;
}

bool Board::HasPawnOn7th()
{
	if (m_boardDescU8[SIDE_TO_MOVE] == WHITE)
	{
		return RANKS[RANK_7] & m_boardDescBB[WP];
	}
	else
	{
		return RANKS[RANK_2] & m_boardDescBB[BP];
	}
}

PieceType Board::GetOpponentLargestPieceType()
{
	Color opponentColor = m_boardDescU8[SIDE_TO_MOVE] ^ COLOR_MASK;

	if (m_boardDescBB[WQ | opponentColor])
	{
		return WQ;
	}

	if (m_boardDescBB[WR | opponentColor])
	{
		return WR;
	}

	if (m_boardDescBB[WB | opponentColor])
	{
		return WB;
	}

	if (m_boardDescBB[WN | opponentColor])
	{
		return WN;
	}

	// we return WP even if the opponent has no pawn, for simplicity
	return WP;
}

bool Board::Is3Fold()
{
	// see whether this position has appeared twice already (at this point, the current hash shouldn't be in the stack)
	// it doesn't matter what order - we are iterating through the whole list anyways
	uint32_t count = 0;
	for (size_t i = 0; i < m_hashStack.GetSize(); ++i)
	{
		if (m_hashStack[i] == m_boardDescBB[HASH])
		{
			++count;

			if (count >= 2)
			{
				return true;
			}
		}
	}

	return false;
}

bool Board::Is2Fold(size_t numMoves)
{
	if (numMoves > m_hashStack.GetSize())
	{
		numMoves = m_hashStack.GetSize();
	}

	for (size_t i = 0; i < numMoves; ++i)
	{
		if (m_hashStack[m_hashStack.GetSize() - 1 - i] == m_boardDescBB[HASH])
		{
			return true;
		}
	}

	return false;
}

bool Board::HasInsufficientMaterial(bool relaxed) const
{
	// if we have any queen or rook or pawn, this is not insufficient
	if (m_boardDescBB[WP] || m_boardDescBB[BP] || m_boardDescBB[WQ] || m_boardDescBB[BQ] || m_boardDescBB[WR] || m_boardDescBB[BR])
	{
		return false;
	}

	auto canWinFunc = [relaxed](const uint64_t &knights, const uint64_t &bishops) -> bool
	{
		if (!knights && !bishops)
		{
			return false;
		}

		// if we don't have knights, we can only win if we have bishops on opposite colours
		if (!knights)
		{
			uint64_t b_on_white = bishops & WHITE_SQUARES;
			uint64_t b_on_black = bishops & BLACK_SQUARES;

			return b_on_white && b_on_black;
		}

		// if we don't have bishops, we can only win if we have at least 3 knights (highly unlikely)
		if (!bishops)
		{
			if (relaxed)
			{
				return PopCount(knights) >= 3;
			}
			else
			{
				// 2 knights cannot force a mate (but can mate with suicidal opponent)
				return PopCount(knights) >= 2;
			}
		}

		// KBN is doable
		return true;
	};

	return !canWinFunc(m_boardDescBB[WN], m_boardDescBB[WB]) && !canWinFunc(m_boardDescBB[BN], m_boardDescBB[BB]);
}

Board::GameStatus Board::GetGameStatus()
{
	MoveList legalMoves;
	GenerateAllLegalMoves<ALL>(legalMoves);

	if (legalMoves.GetSize() == 0)
	{
		if (InCheck())
		{
			if (m_boardDescU8[SIDE_TO_MOVE] == WHITE)
			{
				return BLACK_WINS;
			}
			else
			{
				return WHITE_WINS;
			}
		}
		else
		{
			return STALEMATE;
		}
	}
	else if (HasInsufficientMaterial(false))
	{
		return INSUFFICIENT_MATERIAL;
	}
	else
	{
		return ONGOING;
	}
}

PieceType Board::ApplyMoveSee(PieceType pt, Square from, Square to)
{
	PieceType capturedPiece = m_boardDescU8[to];

	UndoListBB &ulBB = m_undoStackBB.PrePush();
	UndoListU8 &ulU8 = m_undoStackU8.PrePush();

	// no need to clear them because we know exactly how many elements will be on the list

	// we need to update the to-square MB because next ApplyMoveSee call will use it to determine captured piece
	ulU8[0].first = to;
	ulU8[0].second = m_boardDescU8[to];
	m_boardDescU8[to] = pt;

	// we need to update the PT BB because otherwise GenerateSmallestCapture will see it again
	ulBB[0].first = pt;
	ulBB[0].second = m_boardDescBB[pt];
	m_boardDescBB[pt] &= InvBit(from);

	// we need to update occupancy for discovered attacks
	// borrow a space on the undo list for our own occupancy BB
	ulBB[1].second = m_boardDescBB[pt];
	m_seeTotalOccupancy &= InvBit(from);

	m_boardDescU8[SIDE_TO_MOVE] ^= COLOR_MASK;

	return capturedPiece;
}

bool Board::IsSeeEligible(Move mv)
{
	Square to = GetToSquare(mv);

	return m_boardDescU8[to] != EMPTY;
}

void Board::UndoMoveSee()
{
	UndoListBB &ulBB = m_undoStackBB.Top();
	UndoListU8 &ulU8 = m_undoStackU8.Top();

	m_boardDescU8[ulU8[0].first] = ulU8[0].second;
	m_boardDescBB[ulBB[0].first] = ulBB[0].second;
	m_seeTotalOccupancy = ulBB[1].second;

	m_boardDescU8[SIDE_TO_MOVE] ^= COLOR_MASK;

	m_undoStackBB.Pop();
	m_undoStackU8.Pop();
}

bool Board::GenerateSmallestCaptureSee(PieceType &pt, Square &from, Square to)
{
	Color stm = m_boardDescU8[SIDE_TO_MOVE];
	PieceType lastPT = stm == WHITE ? m_seeLastWhitePT : m_seeLastBlackPT;

	uint64_t attackers = 0ULL;

	switch (lastPT)
	{
	case WP:
		attackers = PAWN_ATK[to][stm == WHITE ? 1 : 0] & m_boardDescBB[WP | stm];

		if (attackers)
		{
			pt = WP | stm;
			from = BitScanForward(attackers);
			return true;
		}
		else
		{
			UpdateseeLastPT_(WN);
		}
		// fallthrough
	case WN:
		attackers = KNIGHT_ATK[to] & m_boardDescBB[WN | stm];

		if (attackers)
		{
			pt = WN | stm;
			from = BitScanForward(attackers);
			return true;
		}
		else
		{
			UpdateseeLastPT_(WB);
		}
		// fall through
	case WB:
		attackers = Bmagic(to, m_seeTotalOccupancy) & m_boardDescBB[WB | stm];

		if (attackers)
		{
			pt = WB | stm;
			from = BitScanForward(attackers);
			return true;
		}
		else
		{
			UpdateseeLastPT_(WR);
		}
		// fall through
	case WR:
		attackers = Rmagic(to, m_seeTotalOccupancy) & m_boardDescBB[WR | stm];

		if (attackers)
		{
			pt = WR | stm;
			from = BitScanForward(attackers);
			return true;
		}
		else
		{
			UpdateseeLastPT_(WQ);
		}
		// fall through
	case WQ:
		attackers = Qmagic(to, m_seeTotalOccupancy) & m_boardDescBB[WQ | stm];

		if (attackers)
		{
			pt = WQ | stm;
			from = BitScanForward(attackers);
			return true;
		}
		else
		{
			UpdateseeLastPT_(WK);
		}
		// fall through
	case WK:
		attackers = KING_ATK[to] & m_boardDescBB[WK | stm];

		if (attackers)
		{
			pt = WK | stm;
			from = BitScanForward(attackers);
			return true;
		}
		break;
	default:
		assert(false);
	}

	return false;
}

PieceType Board::GetCapturedPieceType(Move violentMove)
{
	Square to = GetToSquare(violentMove);

	PieceType pieceAtSquare = GetPieceAtSquare(to);

	if (pieceAtSquare != EMPTY)
	{
		return pieceAtSquare;
	}
	else if (GetY(to) == 2 || GetY(to) == 5)
	{
		// if Y is 2 or 5, this is en passant
		return (GetSideToMove() == WHITE) ? BP : WP;
	}
	else
	{
		// this must be a non-capturing promotion
		return EMPTY;
	}
}

uint64_t Board::SpeculateHashAfterMove(Move mv)
{
	uint64_t hash = m_boardDescBB[HASH];

	PieceType pt = GetPieceType(mv);
	Square from = GetFromSquare(mv);
	Square to = GetToSquare(mv);

	hash ^= PIECES_ZOBRIST[from][pt];
	hash ^= PIECES_ZOBRIST[to][pt];

	hash ^= SIDE_TO_MOVE_ZOBRIST;

	PieceType originalPiece = GetPieceAtSquare(to);

	if (originalPiece != EMPTY)
	{
		hash ^= PIECES_ZOBRIST[to][originalPiece];
	}

	return hash;
}

template <PieceType PT>
uint64_t Board::GetAttackers(Square sq) const
{
	uint64_t atkMask = 0;

	if (PT == WK || PT == BK)
	{
		atkMask = KING_ATK[sq];
	}
	else if (PT == WN || PT == BN)
	{
		atkMask = KNIGHT_ATK[sq];
	}
	else if (PT == WB || PT == BB)
	{
		atkMask = Bmagic(sq, m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED] | (1ULL << sq));
	}
	else if (PT == WR || PT == BR)
	{
		atkMask = Rmagic(sq, m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED] | (1ULL << sq));
	}
	else if (PT == WQ || PT == BQ)
	{
		atkMask = Qmagic(sq, m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED] | (1ULL << sq));
	}
	else if (PT == WP)
	{
		atkMask = PAWN_ATK[sq][1]; // to find white attackers, we pretend we are a black attacker
	}
	else if (PT == BP)
	{
		atkMask = PAWN_ATK[sq][0];
	}
	else
	{
		assert(false);
	}

	return atkMask & m_boardDescBB[PT];
}

// instantiate templates
template uint64_t Board::GetAttackers<WK>(Square sq) const;
template uint64_t Board::GetAttackers<WQ>(Square sq) const;
template uint64_t Board::GetAttackers<WR>(Square sq) const;
template uint64_t Board::GetAttackers<WB>(Square sq) const;
template uint64_t Board::GetAttackers<WN>(Square sq) const;
template uint64_t Board::GetAttackers<WP>(Square sq) const;

template uint64_t Board::GetAttackers<BK>(Square sq) const;
template uint64_t Board::GetAttackers<BQ>(Square sq) const;
template uint64_t Board::GetAttackers<BR>(Square sq) const;
template uint64_t Board::GetAttackers<BB>(Square sq) const;
template uint64_t Board::GetAttackers<BN>(Square sq) const;
template uint64_t Board::GetAttackers<BP>(Square sq) const;

void Board::ApplyVariation(const std::vector<Move> &moves)
{
	std::string original = GetFen();
	std::string mvsApplied;

	for (const auto &move : moves)
	{
		mvsApplied += MoveToAlg(move) + ' ';

		bool isLegal = ApplyMove(move);

		if (!isLegal)
		{
			std::cout << original << "\n" << mvsApplied << std::endl;
			throw std::runtime_error(std::string("Illegal move in variation! - ") + MoveToAlg(move) + " FEN: " + GetFen());
		}
	}
}

void Board::ComputeLeastValuableAttackers(PieceType attackers[64], uint8_t numAttackers[64], Color side)
{
	uint64_t kings = m_boardDescBB[WK | side];
	uint64_t queens = m_boardDescBB[WQ | side];
	uint64_t rooks = m_boardDescBB[WR | side];
	uint64_t bishops = m_boardDescBB[WB | side];
	uint64_t knights = m_boardDescBB[WN | side];
	uint64_t pawns = m_boardDescBB[WP | side];

	// initialize everything to empty
	for (Square sq = 0; sq < 64; ++sq)
	{
		attackers[sq] = EMPTY;
		numAttackers[sq] = 0;
	}

	auto updateTableFcn = [&attackers, &numAttackers](PieceType pt, uint64_t bb)
	{
		while (bb)
		{
			Square sq = Extract(bb);
			attackers[sq] = pt;
			++numAttackers[sq];
		}
	};

	uint64_t occupied = m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED];

	// now we start from the most valuable and go to least, and just keep overwriting
	while (kings)
	{
		Square sq = Extract(kings);

		updateTableFcn(WK, KING_ATK[sq]);
	}

	while (queens)
	{
		Square sq = Extract(queens);

		updateTableFcn(WQ, Qmagic(sq, occupied));
	}

	while (rooks)
	{
		Square sq = Extract(rooks);

		updateTableFcn(WR, Rmagic(sq, occupied));
	}

	while (bishops)
	{
		Square sq = Extract(bishops);

		updateTableFcn(WB, Bmagic(sq, occupied));
	}

	while (knights)
	{
		Square sq = Extract(knights);

		updateTableFcn(WN, KNIGHT_ATK[sq]);
	}

	while (pawns)
	{
		Square sq = Extract(pawns);

		updateTableFcn(WP, PAWN_ATK[sq][(side == WHITE) ? 0 : 1]);
	}
}

Optional<Move> Board::GetMoveFromLast(int32_t n)
{
	Optional<Move> ret;

	if (n < static_cast<int32_t>(m_moveStack.GetSize()))
	{
		ret = m_moveStack[m_moveStack.GetSize() - 1 - n];
	}

	return ret;
}

template <Board::MOVE_TYPES MT>
void Board::GenerateAllPseudoLegalMoves_(MoveList &moveList) const
{
	Color sideToMove = m_boardDescU8[SIDE_TO_MOVE];

	GeneratePawnMoves_<MT>(sideToMove, moveList);
	GenerateKnightMoves_<MT>(sideToMove, moveList);
	GenerateBishopMoves_<MT>(sideToMove, moveList);
	GenerateRookMoves_<MT>(sideToMove, moveList);
	GenerateQueenMoves_<MT>(sideToMove, moveList);
	GenerateKingMoves_<MT>(sideToMove, moveList);

#ifdef DEBUG
	if (MT == ALL)
	{
		MoveList mlQuiet;
		MoveList mlViolent;
		GenerateAllPseudoLegalMoves_<QUIET>(mlQuiet);
		GenerateAllPseudoLegalMoves_<VIOLENT>(mlViolent);
		assert((mlQuiet.GetSize() + mlViolent.GetSize()) == moveList.GetSize());
	}
#endif
}

template <Board::MOVE_TYPES MT>
void Board::GenerateKingMoves_(Color color, MoveList &moveList) const
{
	// there can only be one king
#ifdef DEBUG
	assert(PopCount(m_boardDescBB[WK | color]) == 1);
#endif
	PieceType pt = WK | color;

	uint64_t dstMask = 0;
	if (MT == ALL)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}
	else if (MT == VIOLENT)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
	}
	else
	{
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}

	uint64_t kings = m_boardDescBB[pt];
	uint32_t idx = BitScanForward(kings);
	uint64_t dsts = KING_ATK[idx] & dstMask;

	Move mvTemplate = 0;
	SetFromSquare(mvTemplate, idx);
	SetPieceType(mvTemplate, pt);

	while (dsts)
	{
		uint32_t dst = Extract(dsts);

		Move mv = mvTemplate;

		SetToSquare(mv, dst);

		moveList.PushBack(mv);
	}

	// castling
	if (MT != VIOLENT && !m_boardDescU8[IN_CHECK])
	{
		if (pt == WK && m_boardDescU8[E1] == WK)
		{
			if (m_boardDescU8[W_SHORT_CASTLE] &&
				m_boardDescU8[H1] == WR &&
				m_boardDescU8[F1] == EMPTY &&
				m_boardDescU8[G1] == EMPTY &&
				!IsUnderAttack_(F1))
			{
				// we don't have to check current king pos for under attack because we checked that already
				// we don't have to check destination because we are generating pseudo-legal moves
				Move mv = mvTemplate;
				SetCastlingType(mv, MoveConstants::CASTLE_WHITE_SHORT);
				SetToSquare(mv, G1);
				moveList.PushBack(mv);
			}

			if (m_boardDescU8[W_LONG_CASTLE] &&
				m_boardDescU8[A1] == WR &&
				m_boardDescU8[B1] == EMPTY &&
				m_boardDescU8[C1] == EMPTY &&
				m_boardDescU8[D1] == EMPTY &&
				!IsUnderAttack_(D1))
			{
				// we don't have to check current king pos for under attack because we checked that already
				// we don't have to check destination because we are generating pseudo-legal moves
				Move mv = mvTemplate;
				SetCastlingType(mv, MoveConstants::CASTLE_WHITE_LONG);
				SetToSquare(mv, C1);
				moveList.PushBack(mv);
			}
		}
		else if (pt == BK && m_boardDescU8[E8] == BK)
		{
			if (m_boardDescU8[B_SHORT_CASTLE] &&
				m_boardDescU8[H8] == BR &&
				m_boardDescU8[F8] == EMPTY &&
				m_boardDescU8[G8] == EMPTY &&
				!IsUnderAttack_(F8))
			{
				// we don't have to check current king pos for under attack because we checked that already
				// we don't have to check destination because we are generating pseudo-legal moves
				Move mv = mvTemplate;
				SetCastlingType(mv, MoveConstants::CASTLE_BLACK_SHORT);
				SetToSquare(mv, G8);
				moveList.PushBack(mv);
			}

			if (m_boardDescU8[B_LONG_CASTLE] &&
				m_boardDescU8[A8] == BR &&
				m_boardDescU8[B8] == EMPTY &&
				m_boardDescU8[C8] == EMPTY &&
				m_boardDescU8[D8] == EMPTY &&
				!IsUnderAttack_(D8))
			{
				// we don't have to check current king pos for under attack because we checked that already
				// we don't have to check destination because we are generating pseudo-legal moves
				Move mv = mvTemplate;
				SetCastlingType(mv, MoveConstants::CASTLE_BLACK_LONG);
				SetToSquare(mv, C8);
				moveList.PushBack(mv);
			}
		}
	}
}

template void Board::GenerateKingMoves_<Board::QUIET>(Color color, MoveList &moveList) const;
template void Board::GenerateKingMoves_<Board::VIOLENT>(Color color, MoveList &moveList) const;
template void Board::GenerateKingMoves_<Board::ALL>(Color color, MoveList &moveList) const;

template <Board::MOVE_TYPES MT>
void Board::GenerateQueenMoves_(Color color, MoveList &moveList) const
{
	PieceType pt = WQ | color;

	uint64_t dstMask = 0;
	if (MT == ALL)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}
	else if (MT == VIOLENT)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
	}
	else
	{
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}

	uint64_t queens = m_boardDescBB[pt];

	while (queens)
	{
		uint32_t idx = Extract(queens);

		uint64_t dsts = Qmagic(idx, m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]) & dstMask;

		Move mvTemplate = 0;
		SetFromSquare(mvTemplate, idx);
		SetPieceType(mvTemplate, pt);

		while (dsts)
		{
			uint32_t dst = Extract(dsts);

			Move mv = mvTemplate;

			SetToSquare(mv, dst);

			moveList.PushBack(mv);
		}
	}
}

template void Board::GenerateQueenMoves_<Board::QUIET>(Color color, MoveList &moveList) const;
template void Board::GenerateQueenMoves_<Board::VIOLENT>(Color color, MoveList &moveList) const;
template void Board::GenerateQueenMoves_<Board::ALL>(Color color, MoveList &moveList) const;

template <Board::MOVE_TYPES MT>
void Board::GenerateBishopMoves_(Color color, MoveList &moveList) const
{
	PieceType pt = WB | color;

	uint64_t dstMask = 0;
	if (MT == ALL)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}
	else if (MT == VIOLENT)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
	}
	else
	{
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}

	uint64_t bishops = m_boardDescBB[pt];

	while (bishops)
	{
		uint32_t idx = Extract(bishops);

		uint64_t dsts = Bmagic(idx, m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]) & dstMask;

		Move mvTemplate = 0;
		SetFromSquare(mvTemplate, idx);
		SetPieceType(mvTemplate, pt);

		while (dsts)
		{
			uint32_t dst = Extract(dsts);

			Move mv = mvTemplate;

			SetToSquare(mv, dst);

			moveList.PushBack(mv);
		}
	}
}

template void Board::GenerateBishopMoves_<Board::QUIET>(Color color, MoveList &moveList) const;
template void Board::GenerateBishopMoves_<Board::VIOLENT>(Color color, MoveList &moveList) const;
template void Board::GenerateBishopMoves_<Board::ALL>(Color color, MoveList &moveList) const;

template <Board::MOVE_TYPES MT>
void Board::GenerateKnightMoves_(Color color, MoveList &moveList) const
{
	PieceType pt = WN | color;

	uint64_t dstMask = 0;
	if (MT == ALL)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}
	else if (MT == VIOLENT)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
	}
	else
	{
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}

	uint64_t knights = m_boardDescBB[pt];

	while (knights)
	{
		uint32_t idx = Extract(knights);

		uint64_t dsts = KNIGHT_ATK[idx] & dstMask;

		Move mvTemplate = 0;
		SetFromSquare(mvTemplate, idx);
		SetPieceType(mvTemplate, pt);

		while (dsts)
		{
			uint32_t dst = Extract(dsts);

			Move mv = mvTemplate;

			SetToSquare(mv, dst);

			moveList.PushBack(mv);
		}
	}
}

template void Board::GenerateKnightMoves_<Board::QUIET>(Color color, MoveList &moveList) const;
template void Board::GenerateKnightMoves_<Board::VIOLENT>(Color color, MoveList &moveList) const;
template void Board::GenerateKnightMoves_<Board::ALL>(Color color, MoveList &moveList) const;

template <Board::MOVE_TYPES MT>
void Board::GenerateRookMoves_(Color color, MoveList &moveList) const
{
	PieceType pt = WR | color;

	uint64_t dstMask = 0;
	if (MT == ALL)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}
	else if (MT == VIOLENT)
	{
		dstMask |= m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];
	}
	else
	{
		dstMask |= ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	}

	uint64_t rooks = m_boardDescBB[pt];

	while (rooks)
	{
		uint32_t idx = Extract(rooks);

		uint64_t dsts = Rmagic(idx, m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]) & dstMask;

		Move mvTemplate = 0;
		SetFromSquare(mvTemplate, idx);
		SetPieceType(mvTemplate, pt);

		while (dsts)
		{
			uint32_t dst = Extract(dsts);

			Move mv = mvTemplate;

			SetToSquare(mv, dst);

			moveList.PushBack(mv);
		}
	}
}

template void Board::GenerateRookMoves_<Board::QUIET>(Color color, MoveList &moveList) const;
template void Board::GenerateRookMoves_<Board::VIOLENT>(Color color, MoveList &moveList) const;
template void Board::GenerateRookMoves_<Board::ALL>(Color color, MoveList &moveList) const;

template <Board::MOVE_TYPES MT>
void Board::GeneratePawnMoves_(Color color, MoveList &moveList) const
{
	PieceType pt = WP | color;

	uint64_t pawns = m_boardDescBB[pt];

	uint64_t empty = ~(m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]);
	uint64_t enemy = m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)];

	while (pawns)
	{
		uint32_t idx = Extract(pawns);

		Move mvTemplate = 0;
		SetFromSquare(mvTemplate, idx);
		SetPieceType(mvTemplate, pt);

		uint64_t dsts = PAWN_MOVE_1[idx][color == WHITE ? 0 : 1] & empty;

		// if we can move 1 square, try 2 squares
		// we don't have to check whether we are in rank 2 or 7 or not, because entries are 0 otherwise
		dsts |= (dsts != 0ULL) ? (PAWN_MOVE_2[idx][color == WHITE ? 0 : 1] & empty) : 0ULL;

		if (MT == VIOLENT)
		{
			// if we are generating violent moves, only include promotions
			dsts &= RANKS[RANK_1] | RANKS[RANK_8];
		}

		uint64_t captures = PAWN_ATK[idx][color == WHITE ? 0 : 1] & (enemy | m_boardDescBB[EN_PASS_SQUARE]);

		// only add captures if they are promotions in quiet mode
		if (MT == QUIET)
		{
			dsts |= captures & (RANKS[RANK_1] | RANKS[RANK_8]);
		}
		else
		{
			dsts |= captures;
		}

		while (dsts)
		{
			uint32_t dst = Extract(dsts);

			Move mv = 0;

			if (RANK_OF_SQ[dst] & (RANKS[RANK_1] | RANKS[RANK_8]))
			{
				if (MT == QUIET || MT == ALL)
				{
					// under-promotion
					mv = mvTemplate;
					SetToSquare(mv, dst);
					SetPromoType(mv, WR | color);
					moveList.PushBack(mv);

					mv = mvTemplate;
					SetToSquare(mv, dst);
					SetPromoType(mv, WN | color);
					moveList.PushBack(mv);

					mv = mvTemplate;
					SetToSquare(mv, dst);
					SetPromoType(mv, WB | color);
					moveList.PushBack(mv);
				}

				if (MT == VIOLENT || MT == ALL)
				{
					mv = mvTemplate;
					SetToSquare(mv, dst);
					SetPromoType(mv, WQ | color);
					moveList.PushBack(mv);
				}
			}
			else
			{
				mv = mvTemplate;
				SetToSquare(mv, dst);
				moveList.PushBack(mv);
			}
		}
	}
}

template void Board::GeneratePawnMoves_<Board::QUIET>(Color color, MoveList &moveList) const;
template void Board::GeneratePawnMoves_<Board::VIOLENT>(Color color, MoveList &moveList) const;
template void Board::GeneratePawnMoves_<Board::ALL>(Color color, MoveList &moveList) const;

bool Board::IsUnderAttack_(Square sq) const
{
	Color stm = m_boardDescU8[SIDE_TO_MOVE];
	Color enemyColor = stm ^ COLOR_MASK;
	uint64_t allOccupied = m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED];

	if (KING_ATK[sq] & m_boardDescBB[WK | enemyColor])
	{
		return true;
	}

	if (KNIGHT_ATK[sq] & m_boardDescBB[WN | enemyColor])
	{
		return true;
	}

	if (Rmagic(sq, allOccupied) & (m_boardDescBB[WQ | enemyColor] | m_boardDescBB[WR | enemyColor]))
	{
		return true;
	}

	if (Bmagic(sq, allOccupied) & (m_boardDescBB[WQ | enemyColor] | m_boardDescBB[WB | enemyColor]))
	{
		return true;
	}

	if (PAWN_ATK[sq][stm == WHITE ? 0 : 1] & m_boardDescBB[WP | enemyColor])
	{
		return true;
	}

	return false;
}

void Board::UpdateInCheck_()
{
	Color stm = m_boardDescU8[SIDE_TO_MOVE];
	Square kingPos = BitScanForward(m_boardDescBB[WK | stm]);

	m_boardDescU8[IN_CHECK] = IsUnderAttack_(kingPos);
}

void Board::UpdateHashFull_()
{
	uint64_t newHash = 0;

	// first add all pieces
	for (int32_t i = 0; i < 64; ++i)
	{
		if (m_boardDescU8[i] != EMPTY)
		{
			newHash ^= PIECES_ZOBRIST[i][m_boardDescU8[i]];
		}
	}

	if (m_boardDescBB[EN_PASS_SQUARE])
	{
		newHash ^= EN_PASS_ZOBRIST[BitScanForward(m_boardDescBB[EN_PASS_SQUARE])];
	}

	if (m_boardDescU8[W_SHORT_CASTLE])
	{
		newHash ^= W_SHORT_CASTLE_ZOBRIST;
	}

	if (m_boardDescU8[W_LONG_CASTLE])
	{
		newHash ^= W_LONG_CASTLE_ZOBRIST;
	}

	if (m_boardDescU8[B_SHORT_CASTLE])
	{
		newHash ^= B_SHORT_CASTLE_ZOBRIST;
	}

	if (m_boardDescU8[B_LONG_CASTLE])
	{
		newHash ^= B_LONG_CASTLE_ZOBRIST;
	}

	if (m_boardDescU8[SIDE_TO_MOVE] == BLACK)
	{
		newHash ^= SIDE_TO_MOVE_ZOBRIST;
	}

	m_boardDescBB[HASH] = newHash;
}

uint64_t Perft(Board &b, uint32_t depth)
{
	MoveList ml;
	b.GenerateAllLegalMoves<Board::ALL>(ml);

	uint64_t sum = 0;

#ifdef DEBUG
	Board c = b;
#endif

	for (size_t i = 0; i < ml.GetSize(); ++i)
	{
		if (!b.CheckPseudoLegal(ml[i]))
		{
			std::cout << b.MoveToAlg(ml[i]) << std::endl;
			abort();
		}

		if (b.ApplyMove(ml[i]))
		{
			if (depth == 1)
			{
				++sum;
			}
			else
			{
				sum += Perft(b, depth - 1);
			}

			b.UndoMove();

#ifdef DEBUG
			assert(b == c);
#endif
		}
	}

	return sum;
}

uint64_t PerftWithNull(Board &b, uint32_t depth)
{
	MoveList ml;
	b.GenerateAllLegalMoves<Board::ALL>(ml);

	uint64_t sum = 0;

#ifdef DEBUG
	Board c = b;
#endif

	if (!b.InCheck())
	{
		b.MakeNullMove();

		if (depth > 1)
		{
			PerftWithNull(b, depth - 1);
		}

		b.UndoMove();
	}

	for (size_t i = 0; i < ml.GetSize(); ++i)
	{
		if (!b.CheckPseudoLegal(ml[i]))
		{
			std::cout << b.GetFen() << std::endl;
			std::cout << b.MoveToAlg(ml[i]) << std::endl;
			abort();
		}

		b.ApplyMove(ml[i]);

		if (depth == 1)
		{
			++sum;
		}
		else
		{
			sum += Perft(b, depth - 1);
		}

		b.UndoMove();

#ifdef DEBUG
		assert(b == c);
#endif
	}

	return sum;
}

uint64_t DebugPerft(Board &b, uint32_t depth)
{
	double startTime = CurrentTime();
	uint64_t result = Perft(b, depth);
	std::cout << result << std::endl;
	double duration = CurrentTime() - startTime;
	std::cout << "Took: " << duration << " seconds" << std::endl;
	std::cout << (result / duration) << " NPS" << std::endl;

	return result;
}

uint64_t DebugPerftWithNull(Board &b, uint32_t depth)
{
	double startTime = CurrentTime();
	uint64_t result = PerftWithNull(b, depth);
	std::cout << result << std::endl;
	double duration = CurrentTime() - startTime;
	std::cout << "Took: " << duration << " seconds" << std::endl;
	std::cout << (result / duration) << " NPS" << std::endl;

	return result;
}

bool CheckPerftWithNull(std::string fen, uint32_t depth, uint64_t expected)
{
	std::cout << "Checking Perft for " << fen << ", Depth: " << depth << std::endl;
	Board b(fen);
	uint64_t result = DebugPerftWithNull(b, depth);
	if (result != expected)
	{
		std::cout << "Perft check failed for - " << fen << std::endl;
		std::cout << "Expected: " << expected << ", Result: " << result << std::endl;
		return false;
	}

	return true;
}

void DebugRunPerftTests()
{
	if (!CheckPerftWithNull("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324ULL)) { abort(); }
	if (!CheckPerftWithNull("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 5, 193690690ULL)) { abort(); }
	if (!CheckPerftWithNull("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 7, 178633661ULL)) { abort(); }
	if (!CheckPerftWithNull("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 6, 706045033ULL)) { abort(); }
	if (!CheckPerftWithNull("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 6, 706045033ULL)) { abort(); }
	if (!CheckPerftWithNull("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6", 3, 53392ULL)) { abort(); }
	if (!CheckPerftWithNull("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551ULL)) { abort(); }
}
