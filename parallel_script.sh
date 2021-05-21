#!/bin/bash

for par in $(ls /project/samee/minjun/MuSeAM/sharpr/pars/*)
do
	qsub -F ${par} run_pbs_parallel.sh
done
