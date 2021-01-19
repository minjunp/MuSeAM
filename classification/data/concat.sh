#!/bin/bash

pos_fa=$(cat config.csv | awk 'NR==2' | cut -f6 -d',')
neg_fa=$(cat config.csv | awk 'NR==2' | cut -f7 -d',')
combined_fa=$(cat config.csv | awk 'NR==2' | cut -f8 -d',')
readout_name=$(cat config.csv | awk 'NR==2' | cut -f9 -d',')
#pos_fa="E13RACtrlF1_E13RAMutF1_DMR_toppos2000_pos_v2.fa"
#neg_fa="E13RACtrlF1_E13RAMutF1_DMR_toppos2000_neg_v2.fa"
#combined_fa="E13RACtrlF1_E13RAMutF1_DMR_toppos2000_combined.fa"
#readout_name="E13RACtrlF1_E13RAMutF1_DMR_toppos2000.dat"

cat ${pos_fa} ${neg_fa} > ${combined_fa}

# create readout file for classification model
p_len=$(cat ${pos_fa} | wc -l)
p_len2=$(expr $p_len / 2)
n_len=$(cat ${neg_fa} | wc -l)
n_len2=$(expr $n_len / 2)

# wrtie readout file
touch $readout_name
for i in $(seq $p_len2)
do
        echo 1 >> $readout_name
done
for i in $(seq $n_len2)
do
        echo 0 >> $readout_name
done
