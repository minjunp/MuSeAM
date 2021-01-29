import sys
from Bio import SeqIO
from Bio.SeqIO import FastaIO
import pandas as pd

fname = pd.read_csv('config.csv')

def fasta_reformat(fname, output_name):
   with open(output_name, "w") as output_handle:
      fasta_out = FastaIO.FastaWriter(output_handle, wrap=None)
      with open(fname, "r") as f:
         for record in SeqIO.parse(f, "fasta"):
            fasta_out.write_record(record)

fasta_reformat(df.pos_fa[0], df.pos_fa_v2[0])
fasta_reformat(df.neg_fa[0], df.neg_fa_v2[0])
#fasta_reformat("E13RACtrlF1_E13RAMutF1_DMR_toppos2000_pos.fa", "E13RACtrlF1_E13RAMutF1_DMR_toppos2000_pos_v2.fa")
#fasta_reformat("E13RACtrlF1_E13RAMutF1_DMR_toppos2000_neg.fa", "E13RACtrlF1_E13RAMutF1_DMR_toppos2000_neg_v2.fa")
