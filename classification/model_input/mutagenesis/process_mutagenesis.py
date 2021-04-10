import numpy as np
import pandas as pd
import sys

filename = "GRCh38_F9_LDLR.2_SORT1.csv"
df = pd.read_csv(filename)
df_F9 = df[df.Element=='F9']
df_LDLR_2 = df[df.Element=='LDLR.2']
df_SORT1 = df[df.Element=='SORT1']

#print(df)

with open('mutagenesis_peaks.fa', 'r') as file:
    temp = file.read().splitlines()

F9_peak = list(temp[1])
LDLR_2_peak = list(temp[3])
SORT1_peak = list(temp[5])

def get_fasta(df, start_position, peak, peak_info, outputName):
    with open(outputName, 'w') as file:
        file.write(peak_info+"\n")
        file.write("".join(peak)+"\n")
        for i in range(len(df)):
            altered_seq = peak.copy()

            # Get the exact nucleotide in particular location
            idx = df.Position.iloc[i] - start_position
            assert peak[idx] == df.Ref.iloc[i] # Make sure it is same as reference nucleotide

            if df.Alt.iloc[i] == '-':
                altered_seq[idx] = ""
            else:
                altered_seq[idx] = df.Alt.iloc[i]
            file.write(peak_info+"\n")
            file.write("".join(altered_seq)+"\n")

F9_start_position = 139530463
get_fasta(df_F9, F9_start_position, F9_peak, ">chrX-139530462-139530765", 'F9.fa')

LDLR_2_start_position = 11089231
get_fasta(df_LDLR_2, LDLR_2_start_position, LDLR_2_peak, ">chr19-11089230-11089548", 'LDLR_2.fa')

SORT1_start_position = 109274652
get_fasta(df_SORT1, SORT1_start_position, SORT1_peak, ">chr1-109274651-109275251", 'SORT1.fa')
