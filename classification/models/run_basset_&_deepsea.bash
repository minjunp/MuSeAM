#!/bin/bash

nohup python simple_museam_classification.py top_10percent.fa bottom_10percent.fa ../parameters/parameters_museam.txt > ../outs/logs/museam.out &
nohup python deepsea_classification.py top_10percent.fa bottom_10percent.fa ../parameters/parameters_deepsea.txt > ../outs/logs/deepsea.out &
nohup python basset_classification.py top_10percent.fa bottom_10percent.fa ../parameters/parameters_basset.txt > ../outs/logs/basset.out &
