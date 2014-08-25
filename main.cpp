#include <iostream>

#include "magic_moves.h"
#include "board_consts.h"

void Initialize()
{
	initmagicmoves();
	BoardConstsInit();
}

int main(int argc, char **argv)
{
	Initialize();
}
