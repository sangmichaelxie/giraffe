#ifndef BOARD_H
#define BOARD_H

#include <string>

#include "types.h"
#include "board_consts.h"
#include "move.h"

const static std::string DEFAULT_POSITION_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// these definitions are used as indices for the board description arrays
// all aspects of a position are in the 2 arrays (one for bitboards, one for byte fields, including mailbox representation) for ease of undo-ing moves

// 0x0 to 0xd are used for storing piece bitboards
const static uint32_t WHITE_OCCUPIED = 0x6;
const static uint32_t BLACK_OCCUPIED = 0xe;
const static uint32_t EN_PASS_SQUARE = 0x10; // stored as a bitboard since we have 64 bits anyways
const static uint32_t BOARD_HASH = 0x11;

const static uint32_t BOARD_DESC_BB_SIZE = 0x12;

// we also keep a mailbox representation of the board, from 0x0 to 0x3F (64 squares)

// castling rights
const static uint32_t W_SHORT_CASTLE = 0x40;
const static uint32_t W_LONG_CASTLE = 0x41;
const static uint32_t B_SHORT_CASTLE = 0x42;
const static uint32_t B_LONG_CASTLE = 0x43;

const static uint32_t SIDE_TO_MOVE = 0x44;

// half moves clock will overflow at 256 here, but hopefully that won't happen very often
// the engine shouldn't crash regardless (just misevaluate draw)
const static uint32_t HALF_MOVES_CLOCK = 0x45; // number of half moves since last irreversible move (for 50 moves detection)

const static uint32_t IN_CHECK = 0x46; // whether the moving side is in check (this is updated on each board change, so we don't have to recompute many times)

const static uint32_t BOARD_DESC_U8_SIZE = 0x47;

class Board
{
public:
	enum MOVE_TYPES
	{
		QUIET,
		VIOLENT,
		ALL
	};

	typedef FixedVector<std::pair<uint8_t, uint64_t>, 6> UndoListBB; // list of bitboards to revert on undo
	// 6 maximum bitboards (black occupied, white occupied, source piece type, captured piece type, promotion/castling piece type, en passant)

	typedef FixedVector<std::pair<uint8_t, uint8_t>, 8> UndoListU8;
	// For en passant (en passants cannot result in promotion, or reducing castling rights):
	// half moves, in check, source square, destination square, en pass captured square - 5
	// For promotions (capture and non-capture, cannot reduce castling rights):
	// half moves, in check, source square, destination square - 4
	// For regular moves and captures
	// half moves, in check, source square, destination square, changing at most 2 castling rights - 6
	// For castling
	// half moves, in check, king from, king to, rook from, rook to, 2 castling rights - 8


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

	bool InCheck() { return m_boardDescU8[IN_CHECK]; }

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

	uint64_t m_boardDescBB[BOARD_DESC_BB_SIZE];

	// yes, we are using uint64_t to store these u8 values
	// testing shows that this is the fastest
	// testing also shows that using uint8_t in the undo list is the fastest, despite the size mismatch
	uint64_t m_boardDescU8[BOARD_DESC_U8_SIZE];

	GrowableStack<UndoListBB> m_undoStackBB;
	GrowableStack<UndoListU8> m_undoStackU8;
};

uint64_t DebugPerft(std::string fen, uint32_t depth);
void DebugRunPerftTests();

#endif // BOARD_H
