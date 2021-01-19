import sys

def read_fasta_into_list(fasta_file):
    all_seqs = []
    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            line = line.upper()
            if line[0] == '>':
                continue
            else:
                all_seqs.append(line)
    return all_seqs
