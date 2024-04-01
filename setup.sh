#!/bin/sh

if [ $# -eq 0 ]; then
  DATADIR="/datasets/simplebooks"
else
  DATADIR="$1"
fi

mkdir -p ${DATADIR}

cd ${DATADIR}
wget https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip
unzip simplebooks.zip

fasttext  supervised -input ./simplebooks/simplebooks-2/train.txt -output vec -loss ns -dim 8 -lr .5 -epoch 500 -minCount 1
