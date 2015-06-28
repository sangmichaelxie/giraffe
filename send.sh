#!/bin/sh

#rsync -h --progress -z -v -a . batch2:/vol/bitbucket/ml614/tmp/giraffe

rsync -h --progress --exclude 'net.dump' -z -v -a . hpc:/work/ml614/giraffe
