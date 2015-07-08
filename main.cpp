#include <iostream>
#include <fstream>
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
#include "util.h"
#include "movepicker.h"
#include "ann/learn_ann.h"
#include "ann/features_conv.h"
#include "ann/ann_evaluator.h"
#include "learn.h"

#include "Eigen/Dense"

void Initialize()
{
	Eigen::initParallel();

	// set Eigen to use 1 thread because we are doing OpenMP here
	Eigen::setNbThreads(1);

	// disable nested parallelism since we don't need it, and disabling it
	// makes managing number of threads easier
	omp_set_nested(0);

	std::cout << "# Using " << omp_get_max_threads() << " OpenMP thread(s)" << std::endl;

	// turn off IO buffering
	std::cout.setf(std::ios::unitbuf);

	initmagicmoves();
	BoardConstsInit();
	InitializeZobrist();
}

std::string gVersion;

void GetVersion()
{
	std::ifstream verFile("version.txt");

	if (verFile.is_open())
	{
		std::getline(verFile, gVersion);

		std::cout << "# Version: " << gVersion << std::endl;
	}
}

int main(int argc, char **argv)
{
	Initialize();

	GetVersion();

	Backend backend;

	// first we handle special operation modes
	if (argc >= 2 && std::string(argv[1]) == "conv_dataset")
	{
		// operate in special conv_dataset mode, where we convert a bunch of FENs to neural network input format
		if (argc != 5)
		{
			std::cout << "Usage: " << argv[0] << " conv_dataset input_filename output_filename output_features" << std::endl;
			return 1;
		}

		std::vector<FeaturesConv::FeatureDescription> featureDescriptions;

		FeaturesConv::ConvertBoardToNN(Board(), featureDescriptions);

		std::string fen;

		std::ifstream infile(argv[2]);
		std::ofstream outfile(argv[3], std::ofstream::binary);

		{
			std::ofstream featuresfile(argv[4], std::ofstream::binary);

			// first we print the feature descriptions
			for (const auto &fd : featureDescriptions)
			{
				if (fd.featureType == FeaturesConv::FeatureDescription::FeatureType_global)
				{
					featuresfile << 'G' << ' ' << fd.group << std::endl;
				}
				else
				{
					featuresfile << 'S';

					std::set<Square> influences = FeaturesConv::GetInfluences(fd);

					featuresfile << ' ' << influences.size();

					for (const auto &sq : influences)
					{
						featuresfile << ' ' << sq;
					}

					featuresfile << std::endl;
				}
			}
		}

		uint32_t numRows = 0;
		uint32_t numCols = 0;

		// we first write 2*sizeof(uint32_t) of placeholders to be overwritten later
		outfile.write(reinterpret_cast<const char *>(&numRows), sizeof(uint32_t));
		outfile.write(reinterpret_cast<const char *>(&numCols), sizeof(uint32_t));

		while (std::getline(infile, fen))
		{
			Board b(fen);

			std::vector<float> features;

			FeaturesConv::ConvertBoardToNN(b, features);

			if (features.size() != featureDescriptions.size())
			{
				std::cout << "Wrong feature vector size! " << features.size() << " (Expecting: " << featureDescriptions.size() << ")" << std::endl;
				return 1;
			}

			++numRows;
			numCols = features.size();

			outfile.write(reinterpret_cast<const char *>(&features[0]), numCols * sizeof(float));
		}

		outfile.seekp(0);

		outfile.write(reinterpret_cast<const char *>(&numRows), sizeof(uint32_t));
		outfile.write(reinterpret_cast<const char *>(&numCols), sizeof(uint32_t));

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "learn")
	{
		if (argc < 5)
		{
			std::cout << "Usage: " << argv[0] << " learn x_filename y_filename features_filename" << std::endl;
			return 1;
		}

		std::string xFilename = argv[2];
		std::string yFilename = argv[3];
		std::string featuresFilename = argv[4];

		LearnAnn::TrainANNFromFile(xFilename, yFilename, featuresFilename, nullptr, 30);

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "tdl")
	{
		if (argc < 3)
		{
			std::cout << "Usage: " << argv[0] << " tdl positions" << std::endl;
		}

		Learn::TDL(argv[2]);

		return 0;
	}

#ifdef DEBUG
	std::cout << "# Running in debug mode" << std::endl;
#else
	std::cout << "# Running in release mode" << std::endl;
#endif

	ANNEvaluator evaluator("eval.net");
	backend.SetEvaluator(&evaluator);

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
			backend.Quit();
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
			std::cout << backend.DebugEval() << std::endl;
		}
		else if (cmd == "runtests")
		{
			DebugRunSeeTests();
			DebugRunMovePickerTests();
			DebugRunPerftTests();
			std::cout << "All passed!" << std::endl;
		}
		else
		{
			std::cout << "Error (unknown command): " << cmd << std::endl;
		}
	}
}
