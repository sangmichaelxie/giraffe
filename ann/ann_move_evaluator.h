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

#ifndef ANN_MOVE_EVALUATOR_H
#define ANN_MOVE_EVALUATOR_H

#include <iostream>
#include <string>
#include <vector>

#include "move_evaluator.h"

#include "ann_evaluator.h"
#include "ann.h"
#include "learn_ann.h"
#include "features_conv.h"
#include "board.h"

class ANNMoveEvaluator : public MoveEvaluatorIface
{
public:
	// if total node budget is less than this number, switch back to static allocator
	// doesn't make sense to spend more time deciding what to search than actually
	// searching
	const static int64_t MinimumNodeBudget = 10000;

	ANNMoveEvaluator(ANNEvaluator &annEval);

	void Train(const std::vector<std::string> &positions, const std::vector<std::string> &bestMoves);

	void Test(const std::vector<std::string> &positions, const std::vector<std::string> &bestMoves);

	virtual void NotifyBestMove(Board &board, SearchInfo &si, MoveInfoList &list, Move bestMove, size_t movesSearched) override;

	virtual void EvaluateMoves(Board &board, SearchInfo &si, MoveInfoList &list, MoveList &ml);

	virtual void PrintDiag(Board &b) override;

	void Serialize(std::ostream &os);
	void Deserialize(std::istream &is);

private:
	void GenerateMoveConvInfo_(Board &board, MoveList &ml, FeaturesConv::ConvertMovesInfo &convInfo);

	MoveEvalNet m_ann;

	// we need to have an ANN evaluator to generate signatures
	ANNEvaluator &m_annEval;
};

#endif // ANN_MOVE_EVALUATOR_H
