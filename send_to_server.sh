#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 version_name"
	exit 1
fi

VER=$1

TARGET="/vol/bitbucket/ml614/playground/players/$VER"

ssh ic mkdir $TARGET

rsync -avz --exclude '*.epd' --exclude 'Eigen_stable' --exclude '*.o' --exclude '*.a' --exclude '*.d' --exclude 'giraffe' . ic:$TARGET/

echo $VER > version_tmp.txt

scp -r version_tmp.txt ic:$TARGET/version.txt

rm version_tmp.txt
