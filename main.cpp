#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <thread>
#include <mutex>

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
#include "util.h"
#include "ann/learn_ann.h"
#include "ann/features_conv.h"
#include "ann/ann_evaluator.h"
#include "learn.h"
#include "zobrist.h"
#include "gtb.h"
#include "move_evaluator.h"
#include "static_move_evaluator.h"

#include "Eigen/Dense"

std::string gVersion;

void GetVersion()
{
	std::ifstream verFile("version.txt");

	if (verFile.is_open())
	{
		std::getline(verFile, gVersion);

		std::cout << "# Version: " << gVersion << std::endl;
	}
#ifdef HGVERSION
	else
	{
		gVersion = HGVERSION;
		std::cout << "# Version: " << HGVERSION << std::endl;
	}
#endif
}

void InitializeSlow(ANNEvaluator &evaluator, std::mutex &mtx)
{
	std::string initOutput;

	std::ifstream evalNet("eval.net");

	if (evalNet)
	{
		evaluator.Deserialize(evalNet);
	}
	else
	{
		initOutput += "# eval.net not found, ANN evaluator not initialized\n";
	}

	initOutput += GTB::Init();

	std::lock_guard<std::mutex> lock(mtx);
	std::cout << initOutput;
}

void InitializeSlowBlocking(ANNEvaluator &evaluator)
{
	std::mutex mtx;
	InitializeSlow(evaluator, mtx);
}

// fast initialization steps that can be done in main thread
void InitializeFast()
{
	std::cout << "# Using " << omp_get_max_threads() << " OpenMP thread(s)" << std::endl;

	GetVersion();

#ifdef DEBUG
	std::cout << "# Running in debug mode" << std::endl;
#else
	std::cout << "# Running in release mode" << std::endl;
#endif

	Eigen::initParallel();

	// set Eigen to use 1 thread because we are doing OpenMP here
	Eigen::setNbThreads(1);

	// disable nested parallelism since we don't need it, and disabling it
	// makes managing number of threads easier
	omp_set_nested(0);

	// turn off IO buffering
	std::cout.setf(std::ios::unitbuf);

	initmagicmoves();
	BoardConstsInit();
	InitializeZobrist();
}

int main(int argc, char **argv)
{
	InitializeFast();

	Backend backend;

	ANNEvaluator evaluator;

	//backend.SetEvaluator(&Eval::gStaticEvaluator);
	backend.SetEvaluator(&evaluator);

	backend.SetMoveEvaluator(&gStaticMoveEvaluator);

	// first we handle special operation modes
	if (argc >= 2 && std::string(argv[1]) == "tdl")
	{
		InitializeSlowBlocking(evaluator);

		if (argc < 3)
		{
			std::cout << "Usage: " << argv[0] << " tdl positions" << std::endl;
			return 0;
		}

		Learn::TDL(argv[2]);

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "conv")
	{
		InitializeSlowBlocking(evaluator);

		if (argc < 3)
		{
			std::cout << "Usage: " << argv[0] << " conv FEN" << std::endl;
			return 0;
		}

		std::stringstream ss;

		for (int i = 2; i < argc; ++i)
		{
			ss << argv[i] << ' ';
		}

		Board b(ss.str());

		std::vector<FeaturesConv::FeatureDescription> ret;
		FeaturesConv::ConvertBoardToNN(b, ret);

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "bench")
	{
		InitializeSlowBlocking(evaluator);

		Search::SyncSearchDepthLimited(Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), 7, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchDepthLimited(Board("2r2rk1/pp3pp1/b2Pp3/P1Q4p/RPqN2n1/8/2P2PPP/2B1R1K1 w - - 0 1"), 7, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchDepthLimited(Board("8/1nr3pk/p3p1r1/4p3/P3P1q1/4PR1N/3Q2PK/5R2 w - - 0 1"), 7, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchDepthLimited(Board("5R2/8/7r/7P/5RPK/1k6/4r3/8 w - - 0 1"), 7, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchDepthLimited(Board("r5k1/2p2pp1/1nppr2p/8/p2PPp2/PPP2P1P/3N2P1/R3RK2 w - - 0 1"), 7, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchDepthLimited(Board("8/R7/8/1k6/1p1Bq3/8/4NK2/8 w - - 0 1"), 7, backend.GetEvaluator(), backend.GetMoveEvaluator());

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "check_bounds")
	{
		InitializeSlowBlocking(evaluator);

		if (argc < 3)
		{
			std::cout << "Usage: " << argv[0] << " check_bounds <EPD/FEN file>" << std::endl;
			return 0;
		}

		std::ifstream infile(argv[2]);

		if (!infile)
		{
			std::cerr << "Failed to open " << argv[2] << " for reading" << std::endl;
			return 1;
		}

		uint64_t passes = 0;
		uint64_t total = 0;
		float windowSizeTotal = 0.0f;

		std::string fen;
		std::vector<std::string> fens;
		while (std::getline(infile, fen))
		{
			fens.push_back(fen);
		}

		#pragma omp parallel
		{
			auto evaluatorCopy = evaluator;

			#pragma omp for
			for (size_t i = 0; i < fens.size(); ++i)
			{
				Board b(fens[i]);
				float windowSize = 0.0f;
				bool res = evaluatorCopy.CheckBounds(b, windowSize);

				#pragma omp critical(boundCheckAccum)
				{
					if (res)
					{
						++passes;
					}

					++total;

					windowSizeTotal += windowSize;
				}
			}
		}

		std::cout << passes << "/" << total << std::endl;
		std::cout << "Average window size: " << (windowSizeTotal / total) << std::endl;

		return 0;
	}

	// we need a mutex here because InitializeSlow needs to print, and it may decide to
	// print at the same time as the main command loop (if the command loop isn't waiting)
	std::mutex coutMtx;

	coutMtx.lock();

	// do all the heavy initialization in a thread so we can reply to "protover 2" in time
	std::thread initThread(InitializeSlow, std::ref(evaluator), std::ref(coutMtx));

	auto waitForSlowInitFunc = [&initThread, &coutMtx]() { coutMtx.unlock(); initThread.join(); coutMtx.lock(); };

	while (true)
	{
		std::string lineStr;

		coutMtx.unlock();
		std::getline(std::cin, lineStr);
		coutMtx.lock();

		std::stringstream line(lineStr);

		// we set usermove=1, so all commands from xboard start with a unique word
		std::string cmd;
		line >> cmd;

		// this is the list of commands we can process before initialization finished
		if (
			cmd != "xboard" &&
			cmd != "protover" &&
			cmd != "hard" &&
			cmd != "easy" &&
			cmd != "cores" &&
			cmd != "memory" &&
			cmd != "accepted" &&
			cmd != "rejected" &&
			initThread.joinable())
		{
			// wait for initialization to be done
			waitForSlowInitFunc();
		}

		if (cmd == "xboard") {} // ignore since we only support xboard mode anyways
		else if (cmd == "protover")
		{
			int32_t ver;
			line >> ver;

			if (ver >= 2)
			{
				std::string name = "Giraffe";
				if (gVersion != "")
				{
					name += " ";
					name += gVersion;
				}

				std::cout << "feature ping=1 setboard=1 playother=0 san=0 usermove=1 time=1 draw=0 sigint=0 sigterm=0 "
							 "reuse=1 analyze=1 myname=\"" << name << "\" variants=normal colors=0 ics=0 name=0 pause=0 nps=0 "
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
			break;
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
		else if (cmd == "perft")
		{
			int32_t depth;
			line >> depth;
			backend.DebugRunPerft(depth);
		}
		else if (cmd == "perft_with_null")
		{
			// special perft that also tries null moves (but doesn't count them)
			// this is for debugging null make/unmake
			int32_t depth;
			line >> depth;
			backend.DebugRunPerftWithNull(depth);
		}
		else if (cmd == "eval")
		{
			backend.PrintDebugEval();
		}
		else if (cmd == "meval")
		{
			backend.PrintDebugMoveEval();
		}
		else if (cmd == "gtb")
		{
			std::cout << backend.DebugGTB() << std::endl;
		}
		else if (cmd == "runtests")
		{
			//SEE::DebugRunSeeTests();
			DebugRunPerftTests();
			std::cout << "All passed!" << std::endl;
		}
		else if (backend.IsAMove(cmd))
		{
			backend.Usermove(cmd);
		}
		else
		{
			std::cout << "Error (unknown command): " << cmd << std::endl;
		}
	}

	backend.Quit();
	GTB::DeInit();
}
