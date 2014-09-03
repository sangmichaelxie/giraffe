#include <iostream>

#include "magic_moves.h"
#include "board_consts.h"
#include "move.h"
#include "board.h"
#include "eval/eval.h"
#include "see.h"
#include "search.h"

void Initialize()
{
	initmagicmoves();
	BoardConstsInit();
}

int main(int argc, char **argv)
{
#ifdef DEBUG
	std::cout << "# Running in debug mode" << std::endl;
#else
	std::cout << "# Running in release mode" << std::endl;
#endif

	Initialize();

#if 1
	Board b;

	Search::RootSearchContext searchContext;
	searchContext.timeAlloc.normalTime = 10.0;
	searchContext.timeAlloc.maxTime = 20.0;
	searchContext.stopRequest = false;
	searchContext.startBoard = b;
	searchContext.nodeCount = 0;

	Search::AsyncSearch search(searchContext);

	search.Start();

	search.Join();

	return 0;
#endif

#if 0
	DebugRunPerftTests();

	return 0;
#else

	while (true)
	{
		std::string fen;
		std::getline(std::cin, fen);

		Board b(fen);

		Move mv = b.ParseMove("e3e6");

		assert(mv != 0);

		std::cout << StaticExchangeEvaluation(b, mv) << std::endl;

		//b.CheckBoardConsistency();
	}
#endif
}
