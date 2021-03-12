#!/bin/bash

Rscript gkmSVM.R
python process_fasta.py
./concat.sh

