#include <iostream>
#include <iomanip>

#include <cctype>

int Sq(int x, int y) { return y * 8 + x; }
int GetX(int sq) { return sq % 8; }
int GetY(int sq) { return sq / 8; }
uint64_t Bit(int x) { return 1ULL << x; }
int Valid(int x) { return x < 8 && x >= 0; }

void PrintBB(uint64_t b)
{
	for (int y = 7; y >= 0; --y)
	{
		for (int x = 0; x < 8; ++x)
		{
			std::cout << ((b & Bit(Sq(x, y))) ? "1" : "0") << " ";
		}
		
		std::cout << std::endl;
	}
}

uint64_t kingAtk(int sq)
{
	uint64_t ret = 0;
	int x = GetX(sq);
	int y = GetY(sq);
	
	if (Valid(x+1) && Valid(y+1))
	{
		ret |= Bit(Sq(x+1, y+1));
	}
	
	if (Valid(x+1) && Valid(y-1))
	{
		ret |= Bit(Sq(x+1, y-1));
	}
	
	if (Valid(x-1) && Valid(y+1))
	{
		ret |= Bit(Sq(x-1, y+1));
	}
	
	if (Valid(x-1) && Valid(y-1))
	{
		ret |= Bit(Sq(x-1, y-1));
	}
	
	if (Valid(x) && Valid(y+1))
	{
		ret |= Bit(Sq(x, y+1));
	}
	
	if (Valid(x) && Valid(y-1))
	{
		ret |= Bit(Sq(x, y-1));
	}
	
	if (Valid(x+1) && Valid(y))
	{
		ret |= Bit(Sq(x+1, y));
	}
	
	if (Valid(x-1) && Valid(y))
	{
		ret |= Bit(Sq(x-1, y));
	}
	
	return ret;
}

void PrintKingAtks()
{
	std::cout << "const uint64_t KING_ATK[64] =" << std::endl;
	std::cout << "{" << std::endl;
	for (int x = 0; x < 64; ++x)
	{
		if (x % 8 == 0)
		{
			std::cout << "\t";
		}
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << kingAtk(x) << "ULL, ";
		
		if (x % 8 == 7)
		{
			std::cout << std::endl;
		}
	}
	std::cout << "};" << std::endl;
}

uint64_t knightAtk(int sq)
{
	uint64_t ret = 0;
	int x = GetX(sq);
	int y = GetY(sq);
	
	if (Valid(x+1) && Valid(y+2))
	{
		ret |= Bit(Sq(x+1, y+2));
	}
	
	if (Valid(x+1) && Valid(y-2))
	{
		ret |= Bit(Sq(x+1, y-2));
	}
	
	if (Valid(x-1) && Valid(y+2))
	{
		ret |= Bit(Sq(x-1, y+2));
	}
	
	if (Valid(x-1) && Valid(y-2))
	{
		ret |= Bit(Sq(x-1, y-2));
	}
	
	if (Valid(x+2) && Valid(y+1))
	{
		ret |= Bit(Sq(x+2, y+1));
	}
	
	if (Valid(x+2) && Valid(y-1))
	{
		ret |= Bit(Sq(x+2, y-1));
	}
	
	if (Valid(x-2) && Valid(y+1))
	{
		ret |= Bit(Sq(x-2, y+1));
	}
	
	if (Valid(x-2) && Valid(y-1))
	{
		ret |= Bit(Sq(x-2, y-1));
	}
	
	return ret;
}

uint64_t pawnAtk(int sq, int color)
{
	uint64_t ret = 0;
	int x = GetX(sq);
	int y = GetY(sq);
	
	if (color == 0)
	{
		if (Valid(x+1) && Valid(y+1))
		{
			ret |= Bit(Sq(x+1, y+1));
		}
		
		if (Valid(x-1) && Valid(y+1))
		{
			ret |= Bit(Sq(x-1, y+1));
		}
	}
	else
	{
		if (Valid(x+1) && Valid(y-1))
		{
			ret |= Bit(Sq(x+1, y-1));
		}
		
		if (Valid(x-1) && Valid(y-1))
		{
			ret |= Bit(Sq(x-1, y-1));
		}
	}
	
	return ret;
}

uint64_t pawnMove1(int sq, int color)
{
	uint64_t ret = 0;
	int x = GetX(sq);
	int y = GetY(sq);
	
	if (color == 0)
	{
		if (Valid(x) && Valid(y+1))
		{
			ret |= Bit(Sq(x, y+1));
		}
	}
	else
	{
		if (Valid(x) && Valid(y-1))
		{
			ret |= Bit(Sq(x, y-1));
		}
	}
	
	return ret;
}

uint64_t pawnMove2(int sq, int color)
{
	uint64_t ret = 0;
	int x = GetX(sq);
	int y = GetY(sq);
	
	if (color == 0)
	{
		if (Valid(x) && Valid(y+2))
		{
			ret |= Bit(Sq(x, y+2));
		}
	}
	else
	{
		if (Valid(x) && Valid(y-2))
		{
			ret |= Bit(Sq(x, y-2));
		}
	}
	
	return ret;
}

void PrintKnightAtks()
{
	std::cout << "const uint64_t KNIGHT_ATK[64] =" << std::endl;
	std::cout << "{" << std::endl;
	for (int x = 0; x < 64; ++x)
	{
		if (x % 8 == 0)
		{
			std::cout << "\t";
		}
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << knightAtk(x) << "ULL, ";
		
		if (x % 8 == 7)
		{
			std::cout << std::endl;
		}
	}
	std::cout << "};" << std::endl;
}

void PrintPawnAtks()
{
	std::cout << "const uint64_t PAWN_ATK[64][2] =" << std::endl;
	std::cout << "{" << std::endl;
	std::cout << "\t{" << std::endl;
	
	for (int x = 0; x < 64; ++x)
	{
		if (x % 8 == 0)
		{
			std::cout << "\t\t";
		}
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << pawnAtk(x, 0) << "ULL, ";
		
		if (x % 8 == 7)
		{
			std::cout << std::endl;
		}
	}
	
	std::cout << "\t}," << std::endl;
	
	std::cout << "\t{" << std::endl;
	
	for (int x = 0; x < 64; ++x)
	{
		if (x % 8 == 0)
		{
			std::cout << "\t\t";
		}
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << pawnAtk(x, 1) << "ULL, ";
		
		if (x % 8 == 7)
		{
			std::cout << std::endl;
		}
	}
	
	std::cout << "\t}" << std::endl;
	
	std::cout << "};" << std::endl;
}

void PrintPawnMove1()
{
	std::cout << "const uint64_t PAWN_MOVE_1[64][2] =" << std::endl;
	std::cout << "{" << std::endl;
	std::cout << "\t{" << std::endl;
	
	for (int x = 0; x < 64; ++x)
	{
		if (x % 8 == 0)
		{
			std::cout << "\t\t";
		}
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << pawnMove1(x, 0) << "ULL, ";
		
		if (x % 8 == 7)
		{
			std::cout << std::endl;
		}
	}
	
	std::cout << "\t}," << std::endl;
	
	std::cout << "\t{" << std::endl;
	
	for (int x = 0; x < 64; ++x)
	{
		if (x % 8 == 0)
		{
			std::cout << "\t\t";
		}
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << pawnMove1(x, 1) << "ULL, ";
		
		if (x % 8 == 7)
		{
			std::cout << std::endl;
		}
	}
	
	std::cout << "\t}" << std::endl;
	
	std::cout << "};" << std::endl;
}

void PrintPawnMove2()
{
	std::cout << "const uint64_t PAWN_MOVE_2[64][2] =" << std::endl;
	std::cout << "{" << std::endl;
	std::cout << "\t{" << std::endl;
	
	for (int x = 0; x < 64; ++x)
	{
		if (x % 8 == 0)
		{
			std::cout << "\t\t";
		}
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << pawnMove2(x, 0) << "ULL, ";
		
		if (x % 8 == 7)
		{
			std::cout << std::endl;
		}
	}
	
	std::cout << "\t}," << std::endl;
	
	std::cout << "\t{" << std::endl;
	
	for (int x = 0; x < 64; ++x)
	{
		if (x % 8 == 0)
		{
			std::cout << "\t\t";
		}
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << pawnMove2(x, 1) << "ULL, ";
		
		if (x % 8 == 7)
		{
			std::cout << std::endl;
		}
	}
	
	std::cout << "\t}" << std::endl;
	
	std::cout << "};" << std::endl;
}

void PrintSquares()
{
	for (int i = 0; i < 64; ++i)
	{
		std::cout << std::dec;
		std::cout << "const uint32_t " << static_cast<char>(GetX(i) + 'A') << static_cast<char>(GetY(i) + '1') << " = " << i << ";" << std::endl;
	}
}

void PrintFiles()
{
	for (int x = 0; x < 8; ++x)
	{
		uint64_t bb = 0;
		
		for (int y = 0; y < 8; ++y)
		{
			bb |= Bit(Sq(x, y));
		}
		
		//PrintBB(bb);
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << bb << "ULL, " << std::endl;
	}
}

void PrintRanks()
{
	for (int y = 0; y < 8; ++y)
	{
		uint64_t bb = 0;
		
		for (int x = 0; x < 8; ++x)
		{
			bb |= Bit(Sq(x, y));
		}
		
		//PrintBB(bb);
		
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(16) << bb << "ULL, " << std::endl;
	}
}

int main(int argc, char **argv)
{
	PrintRanks();
}
