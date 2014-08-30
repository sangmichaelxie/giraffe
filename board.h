#ifndef BOARD_H
#define BOARD_H

#include <string>

#include "types.h"
#include "board_consts.h"
#include "move.h"

const static std::string DEFAULT_POSITION_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// these definitions are used as indices for the board description array
// all aspects of a position are in the array for ease of undo-ing moves

// 0x0 to 0xd are used for storing piece bitboards
const static uint32_t WHITE_OCCUPIED = 0x6;
const static uint32_t BLACK_OCCUPIED = 0xe;
const static uint32_t EN_PASS_SQUARE = 0x10; // stored as a bitboard since we have 64 bits anyways
const static uint32_t BOARD_HASH = 0x11;

// we also keep a mailbox representation of the board, from 0x12 to 0x51 (64 squares)
// it's a little wasteful to use 64 bits to store piece types, but it's faster to have
// the same size
const static uint32_t MB_BASE = 0x12;
const static uint32_t MB_LAST = 0x51;

// castling rights
const static uint32_t W_SHORT_CASTLE = 0x52;
const static uint32_t W_LONG_CASTLE = 0x53;
const static uint32_t B_SHORT_CASTLE = 0x54;
const static uint32_t B_LONG_CASTLE = 0x55;

const static uint32_t SIDE_TO_MOVE = 0x56;

const static uint32_t HALF_MOVES_CLOCK = 0x57; // number of half moves since last irreversible move (for 50 moves detection)

const static uint32_t IN_CHECK = 0x58; // whether the moving side is in check (this is updated on each board change, so we don't have to recompute many times)

const static uint32_t BOARD_DESC_SIZE = 0x59;

class Board
{
public:
	enum MOVE_TYPES
	{
		QUIET,
		VIOLENT,
		ALL
	};

	typedef FixedVector<std::pair<uint32_t, uint64_t>, 32> UndoList; // list of bitboards to revert on undo

	Board(const std::string &fen);
	Board() : Board(DEFAULT_POSITION_FEN) {}

	void RemovePiece(Square sq);
	void PlacePiece(Square sq, PieceType pt);

	template <MOVE_TYPES MT> void GenerateAllMoves(MoveList &moveList);

#ifdef DEBUG
	// debug function to check consistency between occupied bitboards, piece bitboards, MB, and castling rights
	void CheckBoardConsistency();
#endif

	std::string GetFen(bool omitMoveNums = false) const;

	std::string PrintBoard() const;

	bool InCheck() { return m_boardDesc[IN_CHECK]; }

	// returns whether the move is legal (if not, the move is reverted)
	bool ApplyMove(Move mv);

	void UndoMove();

	std::string MoveToAlg(Move mv);

	bool operator==(const Board &other);

private:
	template <MOVE_TYPES MT> void GenerateKingMoves_(Color color, MoveList &moveList);
	template <MOVE_TYPES MT> void GenerateQueenMoves_(Color color, MoveList &moveList);
	template <MOVE_TYPES MT> void GenerateBishopMoves_(Color color, MoveList &moveList);
	template <MOVE_TYPES MT> void GenerateKnightMoves_(Color color, MoveList &moveList);
	template <MOVE_TYPES MT> void GenerateRookMoves_(Color color, MoveList &moveList);

	// non-quiet only generates captures and promotion to queen
	// quiet only generates non-captures and under-promotions (including captures that result in under-promotion)
	template <MOVE_TYPES MT> void GeneratePawnMoves_(Color color, MoveList &moveList);

	bool IsUnderAttack_(Square sq);
	void UpdateInCheck_();

	uint64_t m_boardDesc[BOARD_DESC_SIZE];
	GrowableStack<UndoList> m_undoStack;
};

uint64_t DebugPerft(std::string fen, uint32_t depth);
void DebugRunPerftTests();

#endif // BOARD_H
