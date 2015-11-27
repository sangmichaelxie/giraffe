import argparse
import numpy as np
import chess
import os 
from chess import pgn

parser = argparse.ArgumentParser(description='Parse pgn file directory to epd.')
parser.add_argument('pgndir', type=str)
parser.add_argument('epddir', type=str)
parser.add_argument('outdir', type=str)
args = parser.parse_args()

iter = 0 
for pgnfile, epdfile in zip(os.listdir(args.pgndir), os.listdir(args.epddir)):
    print pgnfile,epdfile
    iter += 1
    print iter
    with open(os.path.join(args.pgndir, pgnfile), 'r') as pgnf:
        features = []
        offsets = []
        winners = []
        for i, tup in enumerate(pgn.scan_headers(pgnf)):
            offset, header = tup
            offsets.append(offset)
            if header['Result'] == '1-0':
                winner = 'w'
            elif header['Result'] == '0-1':
                winner = 'b'
            else:
                winner = 'none'
            winners.append(winner)
        with open(os.path.join(args.epddir, epdfile), 'r') as epd:
            game_index = 0
            curr_winner = winners[game_index]
            if game_index < len(offsets):
                next_game_offset = offsets[game_index + 1]
            else:
                next_game_offset = float('Inf')
            count = 0
            for j, line in enumerate(epd):
                count += 1
                if j >= next_game_offset:
                    game_index += 1
                    curr_winner = winners[game_index]
                    if game_index < len(offsets):
                        next_game_offset = offsets[game_index + 1]
                    else:
                        next_game_offset = float('Inf')
                    
                toks = line.split()

                #print j, next_game_offset,toks[1],curr_winner 
                if curr_winner != 'none' and toks[1].strip() == curr_winner:
                    features.append(1)
                elif curr_winner != 'none' and toks[1].strip() != curr_winner:
                    features.append(-1)
                else:
                    features.append(0)
            with open(os.path.join(args.outdir, "%s.xie" % epdfile), 'w') as f:
                for feat in features:
                    f.write("%d\n" % feat)
