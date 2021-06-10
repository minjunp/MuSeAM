#!/bin/bash

files=$(ls *.fa)
# order: active - induced - nonactive
touch wt_readout.dat
let value=0
for file in ${files}
do
    num=$(cat $file | wc -l)
    let lineNum=num/2
    for i in $(seq 1 $lineNum)
    do
        echo $value >> wt_readout.dat
    done
    let value=value+1
done

cat $files > sequences.fa
