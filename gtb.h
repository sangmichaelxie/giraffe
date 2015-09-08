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

#ifndef GTB_H
#define GTB_H

#include <utility>

#include "gtb/gtb-probe.h"

#include "types.h"
#include "util.h"
#include "board.h"

// this file is a wrapper for gtb

namespace GTB
{

static const size_t CacheSize = 32*MB;
static const size_t WdlFraction = 96; // use 3/4 of the cache for WDL
static const size_t MaxPieces = 5;

typedef Optional<Score> ProbeResult;

std::string Init(std::string path = "");

ProbeResult Probe(const Board &b);

void DeInit();

}

#endif // GTB_H
