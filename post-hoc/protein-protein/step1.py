import sys
import re
import os
import numpy as np

# Extract coordinate information
def read_header_into_list(fasta_file, drop_N = True):
    header = []
    cur_seq = ""
    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            line = line.upper()
            if line[0] in ['A', 'C', 'G', 'T']:
                header.append(cur_seq)
                cur_seq = ""
                continue
            else:
                cur_seq += line
    if len(cur_seq) > 0:
        if drop_N == True and cur_seq.find('N') >= 0:
            cur_seq = ""
        else:
            header.append(cur_seq)
            cur_seq = ""

    return header

def get_peaks(header): ## take list header
    seqs = []
    for i in range(len(header)):
        seq = header[i]
        if seq[0:2] == '>C':
            str_len = len(seq)
            start = seq.find(':chr',0, str_len)
            end = seq.find('|',0, str_len)
            #end = str_len
            new_seq = seq[start+1:end]
            seqs.append(new_seq)
        else:
            str_len = len(seq)
            start = seq.find('[',0, str_len)
            end = seq.find(']',0, str_len)
            new_seq = seq[start+1:end]
            seqs.append(new_seq)
    return seqs

def process_again(data):
    seqs = []
    for i in range(len(data)):
        seq = data[i]
        if len(seq) > 25:
            #seq[0:7] == '>C:SLEA':

            str_len = len(seq)
            start = seq.find(':chr2',0,str_len)
            #end = seq.find('|',0, str_len)
            end = str_len
            new_seq = seq[13:end]
            seqs.append(new_seq)
        else:
            seqs.append(seq)
    return seqs

def write_peak(peak_name='peaks.txt'):
    headers = read_header_into_list('sequences_control_removed.fa')
    seqs = get_peaks(headers)
    seqs = process_again(seqs)

    text_file = open(peak_name, "w")
    for line in range(len(seqs)):
        text_file.write(seqs[line]+'\n')
    text_file.close()

write_peak()
