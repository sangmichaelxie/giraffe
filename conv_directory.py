import os
import argparse
import pandas as pd
import numpy as np
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--input_ext', default='.epd')
parser.add_argument('--output_ext', default='.xie')
parser.add_argument('--input_path', default='tests/testsuites')
parser.add_argument('--output_path', default='tests/feat_ext_files')

args = parser.parse_args()

for fname in os.listdir(args.input_path):
    if args.input_ext in fname:
        input_full = args.input_path + '/' + fname
        feats_full = args.output_path + '/' + fname + '.feats'
        xie_full = args.output_path + '/' + fname + args.output_ext
        command = "./giraffe conv_file " + input_full + " " + feats_full
        print('Calling ' + command + '...')
        call(command.split())
        feat_file = pd.read_table(feats_full, delimiter=' ').as_matrix()
        np.save(xie_full, feat_file[:,:-1].astype(np.float32))
        command = "rm -rf " + feats_full
        print('Calling ' + command + '...')
        call(command.split())
