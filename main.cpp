#include <iostream>
#include <string>
#include <sstream>

#include <cstdint>

#include "magic_moves.h"
#include "board_consts.h"
#include "move.h"
#include "board.h"
#include "eval/eval.h"
#include "see.h"
#include "search.h"

void Initialize()
{
	// turn off IO buffering
	std::cout.setf(std::ios::unitbuf);

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
#if 0
	enum EngineMode
	{
		EngineMode_force,
		EngineMode_playingWhite,
		EngineMode_playingBlack,
		EngineMode_analyzing
	};

	EngineMode mode;
	Board currentBoard;

	while (true)
	{
		std::string lineStr;
		std::getline(std::cin, lineStr);

		std::stringstream line(lineStr);

		// we set usermove=1, so all commands from xboard start with a unique word
		std::string cmd;
		line >> cmd;

		if (cmd == "xboard") {} // ignore since we only support xboard mode anyways
		else if (cmd == "protover")
		{
			int32_t ver;
			line >> ver;

			if (ver >= 2)
			{
				std::cout << "feature ping=1 setboard=1 playother=0 san=0 usermove=1 time=1 draw=0 sigint=0 sigterm=0 "
							 "reuse=1 analyze=1 myname=\"Giraffe\" variants=normal colors=0 ics=0 name=0 pause=0 nps=0 "
							 "debug=1 memory=0 smp=0 done=1" << std::endl;
			}
		}
		else if (cmd == "accepted") {}
		else if (cmd == "rejected") {}
		else if (cmd == "new")
		{
			// TODO
		}
		else if (cmd == "quit")
		{
			return 0;
		}
		else if (cmd == "random") {}
		else if (cmd == "force")
		{
			// TODO
		}
		else if (cmd == "go")
		{
			// TODO
		}
		else if (cmd == "level")
		{
			// TODO
		}
		else if (cmd == "st")
		{
			// TODO
		}
		else if (cmd == "sd")
		{
			// TODO
		}
		else if (cmd == "time")
		{
			// TODO
		}
		else if (cmd == "otim")
		{
			// TODO
		}
		else if (cmd == "usermove")
		{
			// TODO
		}
		else if (cmd == "?") {}
		else if (cmd == "result")
		{
			// TODO
		}
		else if (cmd == "ping")
		{
			int32_t num;
			line >> num;
			std::cout << "pong " << num << std::endl;
		}
		else if (cmd == "hint") {}
		else if (cmd == "undo")
		{
			// TODO
		}
		else if (cmd == "remove")
		{
			// TODO
		}
		else if (cmd == "hard")
		{
			// TODO
		}
		else if (cmd == "easy")
		{
			// TODO
		}
		else if (cmd == "post")
		{
			// TODO
		}
		else if (cmd == "nopost")
		{
			// TODO
		}
		else if (cmd == "analyze")
		{
			// TODO
		}
		else if (cmd == "computer") {}
		else
		{
			std::cout << "Error (unknown command): " << cmd << std::endl;
		}
	}
#endif

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
