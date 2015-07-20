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

std::string Init();

ProbeResult Probe(const Board &b);

void DeInit();

}

#endif // GTB_H
