#!/usr/bin/env python3
import os 
import chess
import sys

from chess import pgn
import argparse

parser = argparse.ArgumentParser(description='Parse pgn file directory to epd.')
parser.add_argument('pgndir', type=str)
parser.add_argument('out_dir', type=str)
args = parser.parse_args()

iter = 0 
for pgnfile in os.listdir(args.pgndir):
    iter += 1 
    print iter 
    with open(os.path.join(args.pgndir, pgnfile), 'r') as pgn:
        toks = pgnfile.split('.')
        outfile = "%s.epd" % toks[0]
        with open(os.path.join(args.out_dir, outfile), 'w') as epd:
            game = chess.pgn.read_game(pgn)
            while game != None:
                    gameNode = game
                    while len(gameNode.variations):
                            epd.write(gameNode.board().epd(sm = gameNode.variations[0].move) + '\n')
                            gameNode = gameNode.variations[0]
                    
                    game = chess.pgn.read_game(pgn)

