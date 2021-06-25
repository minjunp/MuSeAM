#!/bin/bash

nohup python basset_RNN.py sequences.fa wt_readout.dat ../parameters/parameters_basset_RNN.txt > ../outs/logs/basset_RNN.out &
