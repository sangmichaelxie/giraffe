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

#ifndef EVAL_H
#define EVAL_H

#include <limits>

#include <cstdint>

#include "board.h"
#include "eval_params.h"
#include "evaluator.h"

namespace Eval
{

// returns score for white
Score StaticEvaluate(Board &b, Score lowerBound, Score upperBound);

// returns score for white
Score EvaluateMaterial(const Board &b);

class StaticEvaluator : public EvaluatorIface
{
public:
	Score EvaluateForWhiteImpl(Board &b, Score lowerBound, Score upperBound) override
	{
		(void) lowerBound;
		(void) upperBound;

		//return StaticEvaluate(b, lowerBound, upperBound);
		return EvaluateMaterial(b);
	}
};

extern StaticEvaluator gStaticEvaluator;

}

#endif // EVAL_H
