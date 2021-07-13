#!/bin/bash
#PBS -l walltime=48:00:00,nodes=1:ppn=2
#PBS -l vmem=64GB
#PBS -N mpra
#PBS -o /mount/samee/hali_data/aggr_results_whole_heart/seurat_analysis/syntax_analysis/fimo/pbs_outs
#PBS -e /mount/samee/hali_data/aggr_results_whole_heart/seurat_analysis/syntax_analysis/fimo/pbs_outs

source /etc/profile.d/modules.sh
#module load anaconda3/5.0.1
module load meme
module load bedtools

source activate
export PATH=/home/minjunp/anaconda3/bin:$PATH
export PATH=/home/minjunp/anaconda3:$PATH
conda activate seurat
conda env list

# Background:
echo "pbs home directory: $PBS_O_HOME"
echo "pbs current working directory: $PBS_O_WORKDIR"
echo "pbs local scratch: $TMPDIR"

### JOB EXECUTION ###
cd $PBS_O_WORKDIR
$PBS_O_WORKDIR/run_fimo.sh
