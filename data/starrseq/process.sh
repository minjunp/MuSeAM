#!/bin/bash

filename='GSE151064_starrseq-lncap-constitutivly-active-regions.bed'
#GSE151064_starrseq-lncap-induced-regions.bed
#GSE151064_starrseq-lncap-nonactive-regions.bed

for filename in GSE151064_starrseq-lncap-constitutivly-active-regions.bed GSE151064_starrseq-lncap-induced-regions.bed GSE151064_starrseq-lncap-nonactive-regions.bed
do
  newName=$(echo ${filename} | cut -f1 -d'.')
  cat ${filename} | awk '{print $1"\t"$2"\t"$3}' > ${newName}_v2.bed
done
