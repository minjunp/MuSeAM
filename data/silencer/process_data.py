import pandas as pd
import numpy as np
import sys

filename = 'Supplementary_Table2.xlsx'
df = pd.read_excel(filename)
df = df.dropna()

sorted_df = df.sort_values('log2FC', ascending=False)

top2000 = sorted_df[0:2000].reset_index()
bottom2000 = sorted_df[-2000:].reset_index()

with open('top2000.bed', 'w') as f:
    for i in range(len(top2000)):
        chr = top2000['chr'][i]
        start = top2000['start'][i]
        end = top2000['end'][i]
        f.write(f'{chr}\t{start}\t{end}\n')

with open('bottom2000.bed', 'w') as f:
    for i in range(len(bottom2000)):
        chr = bottom2000['chr'][i]
        start = bottom2000['start'][i]
        end = bottom2000['end'][i]
        f.write(f'{chr}\t{start}\t{end}\n')

print(top2000)
print(bottom2000)
