import sys
import re
import os
import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import hypergeom
from operator import itemgetter, attrgetter

def get_maxpool_filter(dir='maxpool_filter_control_removed'):
    """
    This function get index for all filters
    Parameter: 'dir' is directory of where the maxpool files are located
    """
    directory = dir
    names = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
             name = os.path.join(directory, filename)
             names.append(name)
             continue
        else:
            continue

    names = sorted(names)

    return names

def get_pair_lists():
    filter = get_maxpool_filter()
    list_of_pairs = [(filter[f1], filter[f2]) for f1 in range(len(filter)) for f2 in range(f1+1,len(filter))]
    #print(len(list_of_pairs)) ## number of pairs = 512*511/2

    return list_of_pairs

def get_index(dir='maxpool_index_ver2_control_removed'):
    directory = dir
    names = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
             name = os.path.join(directory, filename)
             names.append(name)
             continue
        else:
            continue

    names = sorted(names)

    return names

def get_pvalue():
    #cs = []
    geoms = []

    maxpool_ind = get_index() # get maxpool index

    list_of_pairs = get_pair_lists() # get the list of pairs
    pvals = []

    #p = 0.5 ## probability of success
    count = 0
    for ind1 in range(160):
        for ind2 in range(160):
            if abs(ind1-ind2) <= 24:
                count += 1
    p = count/(160*160) # p = 0.2828125

    #for i in range(len(list_of_pairs)):
    for i in range(100):
        pair1 = list_of_pairs[i][0]
        pair2 = list_of_pairs[i][1]

        pair1_end = len(pair1)
        pair2_end = len(pair2)

        pair1_start = pair1.find('/filter',0,pair1_end)
        pair2_start = pair2.find('/filter',0,pair2_end)

        pair1_name = pair1[pair1_start+8:pair1_end]
        pair2_name = pair2[pair2_start+8:pair2_end]

        pair1_ind = 'maxpool_index_ver2_control_removed/filter_'+pair1_name
        pair2_ind = 'maxpool_index_ver2_control_removed/filter_'+pair2_name
        filter_1_ind = np.loadtxt(pair1_ind) # get maxpool indices of the first filter
        filter_2_ind = np.loadtxt(pair2_ind)
        #print(filter_1_ind)
        #print(filter_1_ind[0][0])
        #print(type(filter_1_ind[0][0]))

        filter_1 = np.loadtxt(list_of_pairs[i][0])
        filter_2 = np.loadtxt(list_of_pairs[i][1])
        N_xy, T_xy = [0, 0] # N_xy = number of co-occuring filters, T_xy = with distance <= 12
        for j in range(len(filter_1)):
            # both filter have nonzero
            if filter_1[j] != 0 and filter_2[j] != 0:
                N_xy += 1
                # Abs(A_end - B_start) <= 12 or Abs(A_start - B_end) <= 12
                if abs(filter_1_ind[j][0] - filter_2_ind[j][1]) <= 12 or abs(filter_1_ind[j][1] - filter_2_ind[j][0]) <= 12:

                    T_xy += 1
        print(T_xy, N_xy)

        pval = stats.binom_test(T_xy, N_xy, p, 'greater')
        #print(T_xy)
        #print(N_xy)
        pvals.append(pval)
        """
        # previous version
        a, b, c, d = [0,0,0,0]
        for j in range(len(filter_1)):
            if filter_1[j] != 0 and filter_2[j] != 0:

                a += 1
            elif filter_1[j] != 0 and filter_2[j] == 0:
                b += 1
            elif filter_1[j] == 0 and filter_2[j] != 0:
                c += 1
            elif filter_1[j] == 0 and filter_2[j] == 0:
                d += 1
        """
        #print(a,b,c,d)
        #pval = chisquare([a,b,c,d])[1]
        #pval = chi2_contingency([[a,b],[c,d]])[1]
        #geom = hypergeom.sf(a-1, a+b+c+d, a+c, a+b)
        #pval = stats.fisher_exact([[a,b], [c,d]])[1]
        #cs.append(pval)
        #geoms.append(geom)

    #np.savetxt('hg_cooccur.txt', geoms)
    #plt.hist(pvals, bins=20)
    #plt.show()
    return pvals

#pvals = get_pvalue()
#np.savetxt('binom_cooccur.txt', pvals)
"""
sys.exit()

#for i in range(len(list_of_pairs)):
cs = []
geoms = []
#for i in range(50):
for i in range(len(list_of_pairs)):
    filter_1 = np.loadtxt(list_of_pairs[i][0])
    filter_2 = np.loadtxt(list_of_pairs[i][1])
    a, b, c, d = [0,0,0,0]
    for j in range(len(filter_1)):
        if filter_1[j] != 0 and filter_2[j] != 0:
            a += 1
        elif filter_1[j] != 0 and filter_2[j] == 0:
            b += 1
        elif filter_1[j] == 0 and filter_2[j] != 0:
            c += 1
        elif filter_1[j] == 0 and filter_2[j] == 0:
            d += 1
    #print(a,b,c,d)
    #pval = chisquare([a,b,c,d])[1]
    #pval = chi2_contingency([[a,b],[c,d]])[1]
    geom = hypergeom.sf(a-1, a+b+c+d, a+c, a+b)
    #pval = stats.fisher_exact([[a,b], [c,d]])[1]
    #cs.append(pval)
    geoms.append(geom)

np.savetxt('hg_cooccur.txt', geoms)
sys.exit()
np.savetxt('cs_cooccur.txt', cs)

n = 512*511/2
pval_sig = [i for i in cs if i < 0.05/n]

print(len(cs))




###########################



print(list_of_pairs) #
sys.exit()
hg = np.loadtxt('hg_cooccur.txt')

n = 512*511/2
#alpha = 0.0001
alpha = 0.00001
#alpha = 0.05

pval_sig = [i for i in hg if i < alpha/n] # gives pval in one vector

vals = []
ind = []

for i in range(len(hg)): # hypergeom p-values that have all pairs combined
    if hg[i] < alpha/n:
        vals.append(hg[i]) # p-value of selected pair
        ind.append(i) # index of selected pair

pairs = []
zipped_pairs = []
for i in ind:
    pair = list_of_pairs[i]
    pval_of_pair = hg[i]
    #print(pair)
    #print(pval_of_pair)
    #sys.exit()
    comb = [pair, pval_of_pair]
    zipped_pairs.append(comb)

#print(zipped_pairs[0])
alt = []
for i in range(len(zipped_pairs)):
    if zipped_pairs[i][0][0] == 'maxpool_filter/filter_187.txt':
        alt.append([zipped_pairs[i][0][1], zipped_pairs[i][1]])
    if zipped_pairs[i][0][1] == 'maxpool_filter/filter_187.txt':
        alt.append([zipped_pairs[i][0][0], zipped_pairs[i][1]])
res = sorted(alt, key=itemgetter(1)) # co-occurring pairs in the order of significant p-value
print(res[0:10])

sys.exit()

#print(pairs)
all_vals = []
for i in range(len(pairs)):
    all_vals.append(pairs[i][0])
    all_vals.append(pairs[i][1])

print(all_vals)
sys.exit()
values, counts = np.unique(all_vals, return_counts=True)

zipped = zip(values, counts)
#print(zipped)
zipped = sorted(zipped, key=lambda student: student[1], reverse=True)

print(zipped[0:10])
sys.exit()

print(zip(*zipped))

nums, files = zip(*zipped)

print(files)

#print(type(values))
sys.exit()

print(values)
print(counts)

sys.exit()

print(len(np.sort(counts)))


sys.exit()

#for i in range(len(list_of_pairs)):
cs = []
geoms = []
#for i in range(50):
for i in range(len(list_of_pairs)):
    filter_1 = np.loadtxt(list_of_pairs[i][0])
    filter_2 = np.loadtxt(list_of_pairs[i][1])
    a, b, c, d = [0,0,0,0]
    for j in range(len(filter_1)):
        if filter_1[j] != 0 and filter_2[j] != 0:
            a += 1
        elif filter_1[j] != 0 and filter_2[j] == 0:
            b += 1
        elif filter_1[j] == 0 and filter_2[j] != 0:
            c += 1
        elif filter_1[j] == 0 and filter_2[j] == 0:
            d += 1
    #print(a,b,c,d)
    #pval = chisquare([a,b,c,d])[1]
    #pval = chi2_contingency([[a,b],[c,d]])[1]
    geom = hypergeom.sf(a-1, a+b+c+d, a+c, a+b)
    #pval = stats.fisher_exact([[a,b], [c,d]])[1]
    #cs.append(pval)
    geoms.append(geom)

np.savetxt('hg_cooccur.txt', geoms)
sys.exit()
np.savetxt('cs_cooccur.txt', cs)

n = 512*511/2
pval_sig = [i for i in cs if i < 0.05/n]

print(len(cs))
"""
