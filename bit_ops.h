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

#ifndef BIT_OPS_H
#define BIT_OPS_H

#include <cstdint>
#include <cassert>

inline uint64_t Bit(uint32_t shift)
{
    return 1LL << shift;
}

inline uint64_t InvBit(uint32_t shift)
{
    return ~(1LL << shift);
}

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
