#!/bin/bash
mkdir -p outs

for i in $(seq 1 10)
do
python train.py ./data/silencer/all_sequences.fa ./data/silencer/regression_readout.dat parameter.txt > ./outs/${i}.txt
done
