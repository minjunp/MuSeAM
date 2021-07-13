#!/bin/bash

module load meme
module load bedtools

fasta_file="dnase_10k_sequences.fa"

files=$(ls ./shuffled_pwms)

for i in $files
do
	motif_name=$(echo $i | cut -f2 -d".")
	mkdir -p ./liver_enhancer_shuffled/$motif_name
	echo $motif_name

	thr="1e-4"
	fimo --max-stored-scores 1600000 --verbosity 1 --thresh ${thr} --oc ./liver_enhancer_shuffled/${motif_name} ./shuffled_pwms/${i} ${fasta_file}
done
