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
#include "backend.h"
#include "chessclock.h"

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

	Backend backend;

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
			backend.NewGame();
			backend.SetMaxDepth(0);
		}
		else if (cmd == "setboard")
		{
			std::string fen;
			std::getline(line, fen);
			backend.SetBoard(fen);
		}
		else if (cmd == "quit")
		{
			return 0;
		}
		else if (cmd == "random") {}
		else if (cmd == "force")
		{
			backend.Force();
		}
		else if (cmd == "go")
		{
			backend.Go();
		}
		else if (cmd == "level")
		{
			int32_t movesPerPeriod;

			double base;

			// base is a little complicated, because it can either be minutes or minutes:seconds
			// so we read into a string first before figuring out what to do with it
			std::string baseStr;

			double inc;

			line >> movesPerPeriod;
			line >> baseStr;
			line >> inc;

			if (baseStr.find(':') == std::string::npos)
			{
				sscanf(baseStr.c_str(), "%lf", &base);
				base *= 60.0;
			}
			else
			{
				double minutes;
				double seconds;

				sscanf(baseStr.c_str(), "%lf:%lf", &minutes, &seconds);
				base = minutes * 60.0 + seconds;
			}

			ChessClock cc(ChessClock::CONVENTIONAL_INCREMENTAL_MODE, movesPerPeriod, base, inc);

			backend.SetTimeControl(cc);
		}
		else if (cmd == "st")
		{
			double t;
			line >> t;
			backend.SetTimeControl(ChessClock(ChessClock::EXACT_MODE, 0, 0.0, t));
		}
		else if (cmd == "sd")
		{
			int32_t maxDepth;
			line >> maxDepth;
			backend.SetMaxDepth(maxDepth);
		}
		else if (cmd == "time")
		{
			double t;
			line >> t;
			t /= 100.0;

			backend.AdjustEngineTime(t);
		}
		else if (cmd == "otim")
		{
			double t;
			line >> t;
			t /= 100.0;

			backend.AdjustOpponentTime(t);
		}
		else if (cmd == "usermove")
		{
			std::string mv;
			line >> mv;
			backend.Usermove(mv);
		}
		else if (cmd == "?") {}
		else if (cmd == "result")
		{
			backend.NewGame();
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
			backend.Undo(1);
		}
		else if (cmd == "remove")
		{
			backend.Undo(2);
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
			backend.SetShowThinking(true);
		}
		else if (cmd == "nopost")
		{
			backend.SetShowThinking(false);
		}
		else if (cmd == "analyze")
		{
			backend.SetAnalyzing(true);
		}
		else if (cmd == "exit")
		{
			backend.SetAnalyzing(false);
		}
		else if (cmd == "computer") {}
		else if (cmd == "printboard")
		{
			// for debugging, not in xboard protocol
			backend.DebugPrintBoard();
		}
		else
		{
			std::cout << "Error (unknown command): " << cmd << std::endl;
		}
	}
}
