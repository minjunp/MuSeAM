#!/bin/bash
#PBS -l walltime=48:00:00,nodes=2:ppn=2
#PBS -l vmem=64GB
#PBS -N sharpr
#PBS -o /home/minjunp/MuSeAM/pbs_outs
#PBS -e /home/minjunp/MuSeAM/pbs_outs

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
$PBS_O_WORKDIR/iterate.sh
