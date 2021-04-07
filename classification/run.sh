#!/bin/bash

#python train.py ./dnase/top5k/5k_shuffled_combined.fa ./dnase/top5k/readout.dat parameter1.txt > ./logs/museam_shuffled_20epochs.txt
#python train.py ./dnase/top5k/5k_nullseq_combined.fa ./dnase/top5k/readout.dat parameter1.txt > ./logs/museam_nullseq_20epochs.txt

python train.py ./dnase/top5k/5k_shuffled_combined.fa ./dnase/top5k/readout.dat parameter1.txt > ./logs/deepsea_shuffled_20epochs.txt
python train.py ./dnase/top5k/5k_nullseq_combined.fa ./dnase/top5k/readout.dat parameter1.txt > ./logs/deepsea_nullseq_20epochs.txt
