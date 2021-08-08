#!/bin/bash

for par in $(ls /home/minjunp/MuSeAM/pars/*)
do
	qsub -F ${par} run_pbs_parallel.sh
done
