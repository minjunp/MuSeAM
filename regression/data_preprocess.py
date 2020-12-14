import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dinucleotide import mono_to_dinucleotide, dinucleotide_one_hot_encode

class preprocess:
    def __init__(self, fasta_file, readout_file):
        self.fasta_file = fasta_file
        self.readout_file = readout_file

        #self.read_fasta_into_list()
        #self.read_fasta_forward()
        #self.rc_comp2()
        #self.read_readout()
        #self.without_augment()
        #self.augment()
        #self.one_hot_encode()

    def read_fasta_into_list(self, drop_N = True):
    	all_seqs = []
    	cur_seq = ""
    	with open(self.fasta_file, "r") as f:
    		for line in f:
    			line = line.strip()
    			if line == "":
    				continue
    			line = line.upper()
    			if line[0] == '>':
    				if len(cur_seq) > 0:
    					if drop_N == True and cur_seq.find('N') >= 0:
    						cur_seq = ""
    					else:
    						all_seqs.append(cur_seq)
    						cur_seq = ""
    				continue
    			else:
    				cur_seq += line
    	if len(cur_seq) > 0:
    		if drop_N == True and cur_seq.find('N') >= 0:
    			cur_seq = ""
    		else:
    			all_seqs.append(cur_seq)
    			cur_seq = ""

    	return all_seqs

    def read_fasta_forward(self):
        # get sequences and remove other information
        sequences = []
        with open(self.fasta_file, 'r') as f:
            for count, line in enumerate(f, start=1):
                if count % 2 == 0:
                    sequences.append(line)

        # remove /n in sequences
        sequences = [word.strip() for word in sequences]
        return sequences

    # augment the samples with reverse complement
    def rc_comp2(self):

        def rc_comp(seq):
            rc_dict = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
            rc_seq = ''.join([rc_dict[c] for c in seq[::-1]])
            return rc_seq

        seqn = self.read_fasta_into_list()
        all_sequences = []
        for seq in range(len(seqn)):
            all_sequences.append(rc_comp(seqn[seq]))

        #return all_sequences

        return all_sequences

    # to augment on readout data
    def read_readout(self):
        all_readout = []
        with open(self.readout_file, "r") as f:
            all_readout = list(map(float, f.readlines()))

        return all_readout

    def augment(self):
        new_fasta = self.read_fasta_into_list()
        rc_fasta = self.rc_comp2()
        readout = self.read_readout()

        dict = {"new_fasta": new_fasta, "readout":readout, "rc_fasta": rc_fasta}
        return dict

    def without_augment(self):
        new_fasta = self.read_fasta_into_list()
        readout = self.read_readout()

        dict = {"new_fasta": new_fasta, "readout":readout}
        return dict

    def one_hot_encode(self):
        # The LabelEncoder encodes a sequence of bases as a sequence of integers.
        integer_encoder = LabelEncoder()
        # The OneHotEncoder converts an array of integers to a sparse matrix where
        # each row corresponds to one possible value of each feature.
        one_hot_encoder = OneHotEncoder(categories='auto')

        #dict = self.without_augment()
        dict = self.augment()

        forward = []
        reverse = []
        for sequence in dict["new_fasta"]:
            integer_encoded = integer_encoder.fit_transform(list(sequence))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            forward.append(one_hot_encoded.toarray())

        forward = np.stack(forward)
        dict["forward"] = forward

        for sequence in dict["rc_fasta"]:
            integer_encoded = integer_encoder.fit_transform(list(sequence))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            reverse.append(one_hot_encoded.toarray())

        reverse = np.stack(reverse)
        dict["reverse"] = reverse

        return dict

    def dinucleotide_encode(self):
        new_fasta = self.read_fasta_into_list()
        sequences = mono_to_dinucleotide(new_fasta)

        input_features = dinucleotide_one_hot_encode(sequences)

        dict = self.without_augment()
        dict["input_features"] = input_features
        return dict
