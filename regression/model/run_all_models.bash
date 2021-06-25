#!/bin/bash

python basset.py sequences.fa wt_readout.dat ../parameters/parameters_basset.txt > ../outs/logs/basset.out 
python deepsea.py sequences.fa wt_readout.dat ../parameters/parameters_deepsea.txt > ../outs/logs/deepsea.out
python deepsea.py silencer_sequences.fa silencer_readout.dat ../parameters/parameters_deepsea_silencer.txt > ../outs/logs/deepsea_silencer.out
python basset.py silencer_sequences.fa silencer_readout.dat ../parameters/parameters_basset_silencer.txt > ../outs/logs/basset_silencer.out
python basset_RNN.py silencer_sequences.fa silencer_readout.dat ../parameters/parameters_basset_RNN_silencer.txt > ../outs/logs/basset_rnn_silencer.out
python deepsea_RNN.py silencer_sequences.fa silencer_readout.dat ../parameters/parameters_deepsea_RNN_silencer.txt > ../outs/logs/deepsea_rnn_silencer.out