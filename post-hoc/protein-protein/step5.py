import numpy as np
import pandas as pd
from step4 import co_occur_pair
import sys

# we are comparing against bioGRID known interactions
df = pd.read_csv('BIOGRID-ALL-3.5.186.tab2.txt', sep="\t", header=0)
df = df[df.columns[2:6]]

# sig_pairs = co_occur_pair(file_name='./biogrid/process/liver_enhancer/binom_cooccur_with_hindrance_liver_enhancer.txt')
# fileName = './biogrid/process/liver_enhancer/all_pairs_with_hindrance_evalue_10_motifs_liver_enhancer_pval_0.01.txt'

sig_pairs = co_occur_pair(file_name='./biogrid/process/silencer/binom_cooccur_with_hindrance_silencer.txt')
fileName = './biogrid/process/silencer/all_pairs_with_hindrance_evalue_10_motifs_silencer_pval_0.01.txt'

count = 0
col1 = df[df.columns[0]]
col2 = df[df.columns[1]]
pairs = []

for i in range(len(sig_pairs)):
    #print(i)
    pair_a = sig_pairs[i][0]
    pair_b = sig_pairs[i][1]

    pair_a_ind = col1[col1 == pair_a].index
    pair_b_ind = col2[col2 == pair_a].index

    combined_ind = pair_a_ind.append(pair_b_ind)

    for j in combined_ind: #combined index
        if pair_a == pair_b:
            pairs.append([pair_a, pair_b])
            print(pair_a, pair_b)
            count += 1
            break
        row_val = df.iloc[j]
        list_vals = row_val.values.tolist()
        if any(pair_b in str(mystring) for mystring in list_vals):
            pairs.append([pair_a, pair_b])
            print(pair_a, pair_b)
            count += 1
            break

print(f'Total search number is {len(sig_pairs)}')
print(f'Total number of co-occurring pairs is {count}')
print(f'Proportion of co-occurring pairs is {count/len(sig_pairs)}')

with open(fileName, 'w') as file:
    for i in sig_pairs:
        file.write(f'{i[0]}\t{i[1]}\n')

# with open('pairs_with_hindrance_evalue_5_v2.txt', 'w') as file:
#     for i in pairs:
#         file.write(f'{i[0]}\t{i[1]}\n')

# Total search number is 608
# Total number of co-occurring pairs is 27
# Proportion of co-occurring pairs is 0.044407894736842105
