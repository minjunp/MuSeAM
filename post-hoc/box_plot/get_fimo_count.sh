#!/bin/bash


files=$(ls /home/minjunp/MuSeAM/post-hoc/box_plot/liver_enhancer_true_pwms)
touch  liver_enhancer.txt

for i in $files
do
  numLine=$(cat /home/minjunp/MuSeAM/post-hoc/box_plot/liver_enhancer_true_pwms/$i/fimo.tsv | wc -l)
  let count=$numLine-4
  echo $count >> liver_enhancer.txt
done
