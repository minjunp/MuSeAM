#!/bin/bash
mkdir outs

for i in $(seq 1 30)
do
python train.py ./data/silencer/sequences.fa ./data/silencer/wt_readout.dat parameter.txt > ./outs/${i}.txt
done
