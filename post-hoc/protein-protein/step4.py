import numpy as np
import sys
import pandas as pd
import itertools
# from collections import Counter
# from itertools import chain

# we find all significant pairs with distance constraint -- we load those pairs from binomial test
def co_occur_pair(file_name='binom_cooccur.txt'):
    binom_pvals = np.loadtxt(file_name)

    n_tot = 512
    n = n_tot*n_tot/2 ## total number of pairs

    sig_pvals = [i for i in binom_pvals if i < 0.01] # 2272 significant pairs

    list_of_pairs = list(itertools.combinations(range(512), 2))

    zipped = zip(binom_pvals, list_of_pairs) # zip all pvalues with all list of pairs
    zipped = sorted(zipped) # sorting in the order of most significant p-value

    count_same = 0
    count_diff = 0
    pairs_combine = []
    indices = [] # to save all index of particular co-occurring pair -- manual approach

    for i in range(len(sig_pvals)): # We are only looking at top 5360 significant pairs
        pair = zipped[i][1]
        pair_a_num = pair[0] # maxpool output for the FIRST pair
        pair_b_num = pair[1] # maxpool output for the SECOND pair

        # Get TOMTOM info for each filter
        tomtom_a_tsv = pd.read_csv(f'/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_liver_enhancer/motif_files/motif_files/tomtom_outputs_evalue_10/result_{"{0:0=3d}".format(pair_a_num)}/tomtom.tsv', sep='\t')
        # tomtom_a_tsv = pd.read_csv(f'/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_silencer/motif_files/motif_files/tomtom_outputs_evalue_10/result_{"{0:0=3d}".format(pair_a_num)}/tomtom.tsv', sep='\t')

        tomtom_a = tomtom_a_tsv['Target_ID'][0]
        if isinstance(tomtom_a, float): # isnan does not work... so check the instance type
            continue
        tomtom_match_a = tomtom_a[0:tomtom_a.find('_')]

        tomtom_b_tsv = pd.read_csv(f'/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_liver_enhancer/motif_files/motif_files/tomtom_outputs_evalue_10/result_{"{0:0=3d}".format(pair_b_num)}/tomtom.tsv', sep='\t')
        # tomtom_b_tsv = pd.read_csv(f'/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_silencer/motif_files/motif_files/tomtom_outputs_evalue_10/result_{"{0:0=3d}".format(pair_b_num)}/tomtom.tsv', sep='\t')

        tomtom_b = tomtom_b_tsv['Target_ID'][0]
        if isinstance(tomtom_b, float): # isnan does not work... so check the instance type
            continue
        tomtom_match_b = tomtom_b[0:tomtom_b.find('_')]

        #zipped_pair = (tomtom_match_a, tomtom_match_b)
        # get motif number instead
        zipped_pair = (pair_a_num, pair_b_num)

        zipped_pair = sorted(zipped_pair) # sort for the same pair with different order

        pairs_combine.append(zipped_pair)

    return pairs_combine

#sig_pairs = co_occur_pair('./all_data/binom_cooccur.txt')
# print(len(sig_pairs))
