#!/bin/bash

# files=$(ls *_v2.fa)
# # order: active - induced - nonactive
# rm readout.dat
# touch readout.dat
# let value=0
# for file in ${files}
# do
#     num=$(cat $file | wc -l)
#     let lineNum=num/2
#     for i in $(seq 1 $lineNum)
#     do
#         echo $value >> readout.dat
#     done
#     let value=value+1
# done
#
# cat $files > sequences.fa


## New way of generating input data
rm filtered_sequences.fa
rm readout.dat

files=$(ls GSE151064_starrseq-lncap-induced-regions_v2.fa)
# order: active - induced - nonactive
touch readout.dat
let value=0
for file in ${files}
do
    num=$(cat $file | wc -l)
    let lineNum=num/2
    for i in $(seq 1 $lineNum)
    do
        echo $value >> readout.dat
    done
    let value=value+1
done

cat $files > filtered_sequences.fa

files=$(ls GSE151064_starrseq-lncap-nonactive-regions_v2.fa)
# order: active - induced - nonactive
for file in ${files}
do
    let lineNum=500
    for i in $(seq 1 $lineNum)
    do
        echo $value >> readout.dat
    done
done
