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

#ifndef HISTORY_H
#define HISTORY_H

#include <vector>
#include <utility>

#include "types.h"
#include "move.h"
#include "board.h"

class History
{
public:
	History();

	void NotifyCutoff(Move move, NodeBudget nodeBudget);

	void NotifyNoCutoff(Move move, NodeBudget nodeBudget);

	// score is between 0 and 1
	float GetHistoryScore(Move move);

	void NotifyMoveMade();

private:
	// colour, from, to
	uint64_t m_cutoffCounts[2][64][64];
	uint64_t m_nonCutoffCounts[2][64][64];
};

#endif // HISTORY_H
