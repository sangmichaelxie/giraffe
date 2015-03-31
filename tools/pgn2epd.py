#!/usr/bin/env python3

import chess
import sys

from chess import pgn

if len(sys.argv) != 2:
	print("Usage: " + sys.argv[0] + " <PGN file>")
	sys.exit(1)

if sys.argv[1] == '-':
    pgn = sys.stdin
else:
    pgn = open(sys.argv[1])

game = chess.pgn.read_game(pgn)
while game != None:
	
	gameNode = game
	while len(gameNode.variations):
		print(gameNode.board().epd(sm = gameNode.variations[0].move))
		gameNode = gameNode.variations[0]
	
	game = chess.pgn.read_game(pgn)
