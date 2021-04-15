import numpy as np
import pandas as pd
import sys

filename = "GRCh38_F9_LDLR.2_SORT1.csv"
df = pd.read_csv(filename)
df_F9 = df[df.Element=='F9']
df_LDLR_2 = df[df.Element=='LDLR.2']
df_SORT1 = df[df.Element=='SORT1']

"""
    STEP 1
    Method 1: get flanking regions of altered nucleotide
"""
def get_bed(df, outputName):
    chrs = df.Chromosome.values
    chrPos = df.Position.values
    ## Sequence length of 171
    newStart = chrPos - 86 # First nucleotide not included --> subtract 100
    newEnd = chrPos + 85

    with open(outputName, 'w') as file:
        for i in range(len(df)):
            file.write(f'chr{chrs[i]}\t{newStart[i]}\t{newEnd[i]}\n')
#get_bed(df_F9, 'F9.bed')
#get_bed(df_LDLR_2, 'LDLR_2.bed')
#get_bed(df_SORT1, 'SORT1.bed')

"""
    STEP 2
    Description: Change the middle altered nuclotide from Method 1
"""
def change_nucleotide(fastaFile, outputName, refDf):
    with open(fastaFile, 'r') as file:
        temp = file.read().splitlines()

    # Split into two parts (info, sequences)
    seqs = []
    locs = []
    for count, line in enumerate(temp, start=1):
        if count % 2 == 0:
            seqs.append(line.upper()) # Change to upper-case letters
        else:
            locs.append(line)

    refs = refDf.Ref.values
    alts = refDf.Alt.values
    assert len(seqs) == len(refs) # num sequences == num changes

    with open(outputName, 'w') as file:
        for count, line in enumerate(seqs):
            assert line[85] == refs[count]

            temp = list(line)
            if alts[count] == '-':
                temp[85] = ""
            else:
                temp[85] = alts[count]

            # Reference sequence
            file.write(f'{locs[count]}:REF\n')
            file.write(f'{line}\n')
            # Altered sequence
            new_line = "".join(temp)
            file.write(f'{locs[count]}:ALT\n')
            file.write(f'{new_line}\n')

#change_nucleotide('F9.fa', 'F9_processed.fa', df_F9)
#change_nucleotide('LDLR_2.fa', 'LDLR_2_processed.fa', df_LDLR_2)
#change_nucleotide('SORT1.fa', 'SORT1_processed.fa', df_SORT1)

"""
    Method 2: Get altered sequences in same construct (No flanking regions)
"""
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
#get_fasta(df_F9, F9_start_position, F9_peak, ">chrX-139530462-139530765", 'F9.fa')
LDLR_2_start_position = 11089231
#get_fasta(df_LDLR_2, LDLR_2_start_position, LDLR_2_peak, ">chr19-11089230-11089548", 'LDLR_2.fa')
SORT1_start_position = 109274652
#get_fasta(df_SORT1, SORT1_start_position, SORT1_peak, ">chr1-109274651-109275251", 'SORT1.fa')
