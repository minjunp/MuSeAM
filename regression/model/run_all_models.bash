#!/bin/bash

nohup python simple_museam_classification.py top_10percent.fa bottom_10percent.fa ../parameters/parameters_museam_simple.txt > ../outs/logs/museam_simple.out &
nohup python deep_museam_classification.py top_10percent.fa bottom_10percent.fa ../parameters/parameters_museam_deep.txt > ../outs/logs/museam_deep.out &
nohup python deepsea_classification.py top_10percent.fa bottom_10percent.fa ../parameters/parameters_deepsea.txt > ../outs/logs/deepsea.out &
nohup python basset_classification.py top_10percent.fa bottom_10percent.fa ../parameters/parameters_basset.txt > ../outs/logs/basset.out &
