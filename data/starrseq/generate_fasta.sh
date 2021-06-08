#!/bin/bash
touch filtered_seq.fa

nums=$(cat filtered_numbers.txt)

filename='GSE151064_starrseq-lncap-nonactive-regions_v2.fa'
for i in $nums
do
    let i=i*2-1
    let j=i+1

    firstLine=$(awk -v i=$i 'NR==i' $filename)
    secondLine=$(awk -v j=$j 'NR==j' $filename)

    echo $firstLine >> filtered_seq.fa
    echo $secondLine >> filtered_seq.fa
done
