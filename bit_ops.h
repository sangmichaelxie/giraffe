#ifndef BIT_OPS_H
#define BIT_OPS_H

#include <cstdint>
#include <cassert>

// returns the index of the least significant bit set
// undefined if x = 0
inline uint32_t BitScanForward(uint64_t x)
{
#ifdef DEBUG
	assert(x != 0);
#endif
	return __builtin_ctzll(x);
}

// extract the index of the least significant bit, and remove that bit from x
inline uint32_t Extract(uint64_t &x)
{
	uint32_t idx = BitScanForward(x);
	x &= ~(1ULL << idx);
	return idx;
}

inline uint32_t PopCount(uint64_t x)
{
	return __builtin_popcountll(x);
}

#endif // BIT_OPS_H
