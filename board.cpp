#include "board.h"

#include <string>
#include <set>
#include <sstream>
#include <regex>

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "bit_ops.h"
#include "containers.h"
#include "magic_moves.h"
#include "util.h"

namespace
{

std::string SquareToString(Square sq)
{
	char rank = '1' + (sq / 8);
	char file = 'a' + (sq % 8);

	if (sq == 0xff)
	{
		return std::string("-");
	}
	else
	{
		return std::string(&file, 1) + std::string(&rank, 1);
	}
}

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

char PieceTypeToChar(PieceType pt)
{
	switch (pt)
	{
	case WK:
		return 'K';
	case WQ:
		return 'Q';
	case WB:
		return 'B';
	case WN:
		return 'N';
	case WR:
		return 'R';
	case WP:
		return 'P';

	case BK:
		return 'k';
	case BQ:
		return 'q';
	case BB:
		return 'b';
	case BN:
		return 'n';
	case BR:
		return 'r';
	case BP:
		return 'p';

	case EMPTY:
		return ' ';
	}

	return '?';
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
		m_boardDescBB[EN_PASS_SQUARE] = BIT[StringToSquare(std::string(enPassantSq))];
	}

	std::string strCastlingRights(castlingRights);

	m_boardDescU8[W_SHORT_CASTLE] = strCastlingRights.find('K') != std::string::npos;
	m_boardDescU8[W_LONG_CASTLE] = strCastlingRights.find('Q') != std::string::npos;
	m_boardDescU8[B_SHORT_CASTLE] = strCastlingRights.find('k') != std::string::npos;
	m_boardDescU8[B_LONG_CASTLE] = strCastlingRights.find('q') != std::string::npos;

	m_boardDescU8[HALF_MOVES_CLOCK] = halfMoves;

	UpdateInCheck_();

#ifdef DEBUG
	CheckBoardConsistency();
#endif
}

void Board::RemovePiece(Square sq)
{
	m_boardDescBB[m_boardDescU8[sq]] &= INVBIT[sq];

	m_boardDescU8[sq] = EMPTY;

	// faster to reset both than check
	m_boardDescBB[WHITE_OCCUPIED] &= INVBIT[sq];
	m_boardDescBB[BLACK_OCCUPIED] &= INVBIT[sq];
}

void Board::PlacePiece(Square sq, PieceType pt)
{
#ifdef DEBUG
	assert(pt != EMPTY);
	assert(m_boardDescU8[sq] == EMPTY);
#endif

	m_boardDescU8[sq] = pt;
	m_boardDescBB[pt] |= BIT[sq];

	if (GetColor(pt) == WHITE)
	{
		m_boardDescBB[WHITE_OCCUPIED] |= BIT[sq];
	}
	else
	{
		m_boardDescBB[BLACK_OCCUPIED] |= BIT[sq];
	}
}

template <Board::MOVE_TYPES MT>
void Board::GenerateAllMoves(MoveList &moveList) const
{
	Color sideToMove = m_boardDescU8[SIDE_TO_MOVE];

	GenerateKingMoves_<MT>(sideToMove, moveList);
	GenerateQueenMoves_<MT>(sideToMove, moveList);
	GenerateBishopMoves_<MT>(sideToMove, moveList);
	GenerateRookMoves_<MT>(sideToMove, moveList);
	GenerateKnightMoves_<MT>(sideToMove, moveList);
	GeneratePawnMoves_<MT>(sideToMove, moveList);

#ifdef DEBUG
	if (MT == ALL)
	{
		MoveList mlQuiet;
		MoveList mlViolent;
		GenerateAllMoves<QUIET>(mlQuiet);
		GenerateAllMoves<VIOLENT>(mlViolent);
		assert((mlQuiet.GetSize() + mlViolent.GetSize()) == moveList.GetSize());
	}
#endif
}

template <Board::MOVE_TYPES MT>
void Board::GenerateAllLegalMovesSlow(MoveList &moveList) const
{
	MoveList pseudoLegalMoves;
	GenerateAllMoves<MT>(pseudoLegalMoves);

	// we have to make a copy here because this function is const
	Board boardCopy = *this;

	for (size_t i = 0; i < pseudoLegalMoves.GetSize(); ++i)
	{
		if (boardCopy.ApplyMove(pseudoLegalMoves[i]))
		{
			moveList.PushBack(pseudoLegalMoves[i]);
			boardCopy.UndoMove();
		}
	}
}

#ifdef DEBUG
void Board::CheckBoardConsistency()
{
	for (uint32_t sq = 0; sq < 64; ++sq)
	{
		PieceType pt = m_boardDescU8[sq];
		if (pt == EMPTY)
		{
			for (uint32_t i = 0; i < NUM_PIECETYPES; ++i)
			{
				assert(!(m_boardDescBB[PIECE_TYPE_INDICES[i]] & BIT[sq]));
			}

			assert(!(m_boardDescBB[WHITE_OCCUPIED] & BIT[sq]));
			assert(!(m_boardDescBB[BLACK_OCCUPIED] & BIT[sq]));
		}
		else
		{
			if (GetColor(pt) == WHITE)
			{
				assert(m_boardDescBB[WHITE_OCCUPIED] & BIT[sq]);
				assert(!(m_boardDescBB[BLACK_OCCUPIED] & BIT[sq]));
			}
			else
			{
				assert(m_boardDescBB[BLACK_OCCUPIED] & BIT[sq]);
				assert(!(m_boardDescBB[WHITE_OCCUPIED] & BIT[sq]));
			}

			for (uint32_t i = 0; i < NUM_PIECETYPES; ++i)
			{
				if (PIECE_TYPE_INDICES[i] != pt)
				{
					assert(!(m_boardDescBB[PIECE_TYPE_INDICES[i]] & BIT[sq]));
				}
			}

			assert(m_boardDescBB[pt] & BIT[sq]);
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
}
#endif

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

	return ss.str();
}

bool Board::ApplyMove(Move mv)
{
#define MOVE_PIECE(pt, from, to) \
	m_boardDescBB[pt] ^= BIT[from] | BIT[to]; \
	m_boardDescU8[from] = EMPTY; \
	m_boardDescU8[to] = pt;
#define REMOVE_PIECE(pt, sq) \
	m_boardDescBB[pt] &= INVBIT[sq]; \
	m_boardDescU8[sq] = EMPTY;
#define PLACE_PIECE(pt, sq) \
	m_boardDescBB[pt] |= BIT[sq]; \
	m_boardDescU8[sq] = pt;
#define REPLACE_PIECE(pt_old, pt_new, sq) \
	m_boardDescBB[pt_old] &= INVBIT[sq]; \
	m_boardDescBB[pt_new] |= BIT[sq]; \
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

			MOVE_PIECE(WK, E1, G1);
			MOVE_PIECE(WR, H1, F1);
			ulU8.PushBack(std::make_pair(W_SHORT_CASTLE, m_boardDescU8[W_SHORT_CASTLE]));
			ulU8.PushBack(std::make_pair(W_LONG_CASTLE, m_boardDescU8[W_LONG_CASTLE]));
			m_boardDescU8[W_SHORT_CASTLE] = 0;
			m_boardDescU8[W_LONG_CASTLE] = 0;
			m_boardDescBB[WHITE_OCCUPIED] ^= BIT[E1] | BIT[G1] | BIT[H1] | BIT[F1];
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

			MOVE_PIECE(WK, E1, C1);
			MOVE_PIECE(WR, A1, D1);
			ulU8.PushBack(std::make_pair(W_SHORT_CASTLE, m_boardDescU8[W_SHORT_CASTLE]));
			ulU8.PushBack(std::make_pair(W_LONG_CASTLE, m_boardDescU8[W_LONG_CASTLE]));
			m_boardDescU8[W_SHORT_CASTLE] = 0;
			m_boardDescU8[W_LONG_CASTLE] = 0;
			m_boardDescBB[WHITE_OCCUPIED] ^= BIT[E1] | BIT[C1] | BIT[A1] | BIT[D1];
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

			MOVE_PIECE(BK, E8, G8);
			MOVE_PIECE(BR, H8, F8);
			ulU8.PushBack(std::make_pair(B_SHORT_CASTLE, m_boardDescU8[B_SHORT_CASTLE]));
			ulU8.PushBack(std::make_pair(B_LONG_CASTLE, m_boardDescU8[B_LONG_CASTLE]));
			m_boardDescU8[B_SHORT_CASTLE] = 0;
			m_boardDescU8[B_LONG_CASTLE] = 0;
			m_boardDescBB[BLACK_OCCUPIED] ^= BIT[E8] | BIT[G8] | BIT[H8] | BIT[F8];
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

			MOVE_PIECE(BK, E8, C8);
			MOVE_PIECE(BR, A8, D8);
			ulU8.PushBack(std::make_pair(B_SHORT_CASTLE, m_boardDescU8[B_SHORT_CASTLE]));
			ulU8.PushBack(std::make_pair(B_LONG_CASTLE, m_boardDescU8[B_LONG_CASTLE]));
			m_boardDescU8[B_SHORT_CASTLE] = 0;
			m_boardDescU8[B_LONG_CASTLE] = 0;
			m_boardDescBB[BLACK_OCCUPIED] ^= BIT[E8] | BIT[C8] | BIT[A8] | BIT[D8];
		}
	}
	else if ((pt == WP || pt == BP) && BIT[to] == currentEp) // en passant
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

			MOVE_PIECE(WP, from, to);
			REMOVE_PIECE(BP, to - 8);
			m_boardDescBB[WHITE_OCCUPIED] ^= (BIT[from] | BIT[to]);
			m_boardDescBB[BLACK_OCCUPIED] ^= BIT[to - 8];
		}
		else
		{
			ulU8.PushBack(std::make_pair(from, m_boardDescU8[from]));
			ulU8.PushBack(std::make_pair(to, m_boardDescU8[to]));
			ulU8.PushBack(std::make_pair(to + 8, m_boardDescU8[to + 8]));
			ulBB.PushBack(std::make_pair(WP, m_boardDescBB[WP]));
			ulBB.PushBack(std::make_pair(BP, m_boardDescBB[BP]));
			ulBB.PushBack(std::make_pair(BLACK_OCCUPIED, m_boardDescBB[BLACK_OCCUPIED]));

			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED, m_boardDescBB[WHITE_OCCUPIED]));
			ulBB.PushBack(std::make_pair(BLACK_OCCUPIED, m_boardDescBB[BLACK_OCCUPIED]));

			MOVE_PIECE(BP, from, to);
			REMOVE_PIECE(WP, to + 8);
			m_boardDescBB[BLACK_OCCUPIED] ^= (BIT[from] | BIT[to]);
			m_boardDescBB[WHITE_OCCUPIED] ^= BIT[to + 8];
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

			REMOVE_PIECE(pt, from);
			REPLACE_PIECE(m_boardDescU8[to], pt, to);

			m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)] ^= BIT[to];
			m_boardDescBB[WHITE_OCCUPIED | color] ^= BIT[to] | BIT[from];
		}
		else if (!isPromotion && !isCapture)
		{
			ulU8.PushBack(std::make_pair(from, m_boardDescU8[from]));
			ulU8.PushBack(std::make_pair(to, m_boardDescU8[to]));
			ulBB.PushBack(std::make_pair(pt, m_boardDescBB[pt]));

			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED | color, m_boardDescBB[WHITE_OCCUPIED | color]));

			MOVE_PIECE(pt, from, to);
			m_boardDescBB[WHITE_OCCUPIED | color] ^= BIT[to] | BIT[from];
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

			REMOVE_PIECE(pt, from);
			REPLACE_PIECE(m_boardDescU8[to], promoType, to);

			m_boardDescBB[WHITE_OCCUPIED | (color ^ COLOR_MASK)] ^= BIT[to];
			m_boardDescBB[WHITE_OCCUPIED | color] ^= BIT[to] | BIT[from];
		}
		else // !isCapture && isPromotion
		{
			ulU8.PushBack(std::make_pair(from, m_boardDescU8[from]));
			ulU8.PushBack(std::make_pair(to, m_boardDescU8[to]));
			ulBB.PushBack(std::make_pair(pt, m_boardDescBB[pt]));
			ulBB.PushBack(std::make_pair(promoType, m_boardDescBB[promoType]));

			ulBB.PushBack(std::make_pair(WHITE_OCCUPIED | color, m_boardDescBB[WHITE_OCCUPIED | color]));

			REMOVE_PIECE(pt, from);
			PLACE_PIECE(promoType, to);
			m_boardDescBB[WHITE_OCCUPIED | color] ^= BIT[to] | BIT[from];
		}

		// check for pawn move (update ep)
		if (isPawnDoubleMove)
		{
			// this was saved to undo list earlier already
			m_boardDescBB[EN_PASS_SQUARE] = PAWN_MOVE_1[from][pt == WP ? 0 : 1];
		}

		// update castling rights
		if (m_boardDescU8[W_SHORT_CASTLE] && (pt == WK || (pt == WR && from == H1) || (to == H1)))
		{
			ulU8.PushBack(std::make_pair(W_SHORT_CASTLE, m_boardDescU8[W_SHORT_CASTLE]));
			m_boardDescU8[W_SHORT_CASTLE] = 0;
		}

		if (m_boardDescU8[W_LONG_CASTLE] && (pt == WK || (pt == WR && from == A1) || (to == A1)))
		{
			ulU8.PushBack(std::make_pair(W_LONG_CASTLE, m_boardDescU8[W_LONG_CASTLE]));
			m_boardDescU8[W_LONG_CASTLE] = 0;
		}

		if (m_boardDescU8[B_SHORT_CASTLE] && (pt == BK || (pt == BR && from == H8) || (to == H8)))
		{
			ulU8.PushBack(std::make_pair(B_SHORT_CASTLE, m_boardDescU8[B_SHORT_CASTLE]));
			m_boardDescU8[B_SHORT_CASTLE] = 0;
		}

		if (m_boardDescU8[B_LONG_CASTLE] && (pt == BK || (pt == BR && from == A8) || (to == A8)))
		{
			ulU8.PushBack(std::make_pair(B_LONG_CASTLE, m_boardDescU8[B_LONG_CASTLE]));
			m_boardDescU8[B_LONG_CASTLE] = 0;
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
			std::cout << "Duplicate undo entry in BB undo list found! - " << ulBB[i].first << std::endl;
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

		m_undoStackBB.Pop();
		m_undoStackU8.Pop();

		return false;
	}

	// no need to store this
	m_boardDescU8[SIDE_TO_MOVE] = m_boardDescU8[SIDE_TO_MOVE] ^ COLOR_MASK;

	UpdateInCheck_(); // this is for the new side

	return true;

#undef MOVE_PIECE
#undef REMOVE_PIECE
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

	m_undoStackBB.Pop();
	m_undoStackU8.Pop();
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
	GenerateAllLegalMovesSlow<ALL>(moveList);

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

	return 0;
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
	m_boardDescBB[pt] &= INVBIT[from];

	// we need to update occupancy for discovered attacks
	// borrow a space on the undo list for our own occupancy BB
	ulBB[1].second = m_boardDescBB[pt];
	m_seeTotalOccupancy &= INVBIT[from];

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
		attackers = PAWN_ATK[to][stm == WHITE ? 0 : 1] & m_boardDescBB[WP | stm];

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
		[[clang::fallthrough]];
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
		[[clang::fallthrough]];
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
		[[clang::fallthrough]];
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
		[[clang::fallthrough]];
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
		[[clang::fallthrough]];
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

uint64_t Perft(Board &b, uint32_t depth)
{
	MoveList ml;
	b.GenerateAllMoves<Board::ALL>(ml);

	uint64_t sum = 0;

#ifdef DEBUG
	Board c = b;
#endif

	for (size_t i = 0; i < ml.GetSize(); ++i)
	{
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

bool CheckPerft(std::string fen, uint32_t depth, uint64_t expected)
{
	std::cout << "Checking Perft for " << fen << ", Depth: " << depth << std::endl;
	Board b(fen);
	uint64_t result = DebugPerft(b, depth);
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
	if (!CheckPerft("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324ULL)) { abort(); }
	if (!CheckPerft("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 5, 193690690ULL)) { abort(); }
	if (!CheckPerft("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 7, 178633661ULL)) { abort(); }
	if (!CheckPerft("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 6, 706045033ULL)) { abort(); }
	if (!CheckPerft("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 6, 706045033ULL)) { abort(); }
	if (!CheckPerft("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6", 3, 53392ULL)) { abort(); }
	if (!CheckPerft("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551ULL)) { abort(); }
}
