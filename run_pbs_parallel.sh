#!/bin/bash
#PBS -l walltime=48:00:00,nodes=2:ppn=2
#PBS -l vmem=32GB
#PBS -N sharpr
#PBS -o /project/samee/minjun/MuSeAM/sharpr/pbs_outs
#PBS -e /project/samee/minjun/MuSeAM/sharpr/pbs_outs

source /etc/profile.d/modules.sh
#module load anaconda3/5.0.1
source activate
export PATH=/home/minjunp/anaconda3/bin:$PATH
export PATH=/home/minjunp/anaconda3:$PATH
conda activate museam

# Background:
echo "pbs home directory: $PBS_O_HOME"
echo "pbs current working directory: $PBS_O_WORKDIR"
echo "pbs local scratch: $TMPDIR"

### JOB EXECUTION ###
cd $PBS_O_WORKDIR
name=$(echo $1 | cut -f1 -d'.' | cut -f2 -d'_')
python train.py sequences.fa wt_readout.dat $1 > ./outs/${name}.txt
