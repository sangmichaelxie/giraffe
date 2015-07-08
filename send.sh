#!/bin/sh

#rsync -h --progress --exclude 'eval.net' --exclude 'trainingResults/*' -z -v -a . batch2:/vol/bitbucket/ml614/tmp/giraffe

#rsync -h --progress --exclude 'eval.net' --exclude 'trainingResults/*' -z -v -a . hpc:/work/ml614/giraffe

rsync -h --progress --exclude 'eval.net' --exclude 'trainingResults/*' -z -v -a . amazon_compute:/data/giraffe
