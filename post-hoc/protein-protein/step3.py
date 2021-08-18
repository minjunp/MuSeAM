import sys
import re
import os
import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from glob import glob
import keras

# null_prob = 0.1551 # Liver enhancer
# null_prob = 0.1306 # Silencer
null_prob = 0.1314 # DNase I

# generate every pairs without duplicates
list_of_pairs = list(itertools.combinations(range(512), 2))

##### Get relu outputs ######
# relu_output = np.load('./all_data/liver_enhancer_relu.npy') #shape: (2236, 320, 512)
# relu_output = np.load('./all_data/silencer_relu.npy') #shape: (7232, 378, 512)
relu_output = np.load('./all_data/dnase_relu.npy') #shape: (10200, 376, 512)

maxpool_length = int(relu_output.shape[1]/2)
output_name = 'binom_cooccur_with_hindrance_dnase.txt'

pvals = []
for i in range(len(list_of_pairs)):
    print(i)
    element_a = list_of_pairs[i][0]
    element_b = list_of_pairs[i][1]

    filter_a_fwd = relu_output[:, 0:maxpool_length, element_a]
    filter_a_rc = relu_output[:, maxpool_length:, element_a][:,::-1] # reverse order on ReLU output

    filter_b_fwd = relu_output[:, 0:maxpool_length, element_b]
    filter_b_rc = relu_output[:, maxpool_length:, element_b][:,::-1]

    # Add two vectors (See if they are non-zero)
    filter_a_added = np.add(filter_a_fwd, filter_a_rc)
    filter_b_added = np.add(filter_b_fwd, filter_b_rc)
    # print(filter_a_added.shape) # (2236, 160)

    n_tot = 0
    n_cooccur = 0
    for i in range(filter_a_added.shape[0]):
        filter_a_nonzero_index = np.where(filter_a_added[i,:]>0)
        filter_b_nonzero_index = np.where(filter_b_added[i,:]>0)

        if (filter_a_added[i,:]>0).any() or (filter_b_added[i,:]>0).any():
            n_tot += 1
            # Loop over if there is occurrence
            for ind_a in filter_a_nonzero_index[0]:
                for ind_b in filter_b_nonzero_index[0]:
                    # range_a = range(int(ind_a)-12, int(ind_a)+12)
                    range_a_left = range(int(ind_a)-23, int(ind_a)-12)
                    range_a_right = range(int(ind_a)+12, int(ind_a)+23)

                    if (ind_b in range_a_left) or (ind_b in range_a_right):
                        n_cooccur += 1
                        break
                    else:
                        continue
                break # break twice to move to the next sequence

    pval = stats.binom_test(n_cooccur, n_tot, null_prob, 'greater')
    pvals.append(pval)

np.savetxt(output_name, pvals)
