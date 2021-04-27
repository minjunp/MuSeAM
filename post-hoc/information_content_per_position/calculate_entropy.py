import sys
import numpy as np
import find_consensus_seqs
import os
import matplotlib.pyplot as plt
from scipy import stats


font = {'family' : 'Arial',
        #'weight' : 'bold',
        'size'   : 10}

# Get the all the lines in file in a list
list_lines = list()
with open ("HOCOMOCOv11_full_HUMAN_mono_meme_format.meme", "r") as f:
    for line in f:
        list_lines.append(line.strip())

motif_num = 0
hocomoco_entropy = []

for idx in range(len(list_lines)):
    line = str(list_lines[idx])
    vals = []
    if line.startswith('letter'):
        motif_list = []

        while str(list_lines[idx+1])[0:3] != 'URL':
            motif_list.append(list_lines[idx+1])
            idx = idx + 1

        for i in range(len(motif_list)):
            val = motif_list[i].split('\t')
            vals.append(val)

        vals = np.asarray(vals, dtype=np.float32) # numpy array

        motif_entropy = find_consensus_seqs.compute_entropy(vals)

        motif_entropy_normalized = np.divide(motif_entropy, len(vals))
        hocomoco_entropy.append(motif_entropy_normalized)

dir = '../../saved_model/MuSeAM_regression/motif_original'
file_names = []
for filename in os.listdir(dir):
    name = os.path.join(dir, filename)
    file_names.append(name)
    file_names = sorted(file_names)

learned_filters_entropy = []
for f in file_names:
    vals = np.loadtxt(f)
    alpha = 120
    before_alpha = np.multiply(vals, alpha) # divided by alpha
    before_transform_vals = np.multiply([0.295, 0.205, 0.205, 0.295] ,np.exp(before_alpha)) # transform to before multinomial function

    lf_entropy = find_consensus_seqs.compute_entropy(before_transform_vals) # learned filters entropy
    lf_entropy_normalized = np.divide(lf_entropy, len(vals))
    learned_filters_entropy.append(lf_entropy_normalized)

pval_less = stats.ks_2samp(hocomoco_entropy, learned_filters_entropy, alternative='less')
pval_greater = stats.ks_2samp(hocomoco_entropy, learned_filters_entropy, alternative='greater')

#pval2 = stats.kstest(hocomoco_entropy, learned_filters_entropy, alternative='less')
print('ks 2sample with one-tailed (alternative = less) is:')
print(pval_less)
print('')
print('ks 2sample with one-tailed (alternative = greater) is:')
print(pval_greater)

# figure related code
fig, ax = plt.subplots()

bp1 = ax.boxplot(hocomoco_entropy, positions = [1], widths = 0.3)
bp2 = ax.boxplot(learned_filters_entropy, positions = [2], widths = 0.3)

#ax.set_title('Entropy per position of each motif')
ax.set_ylabel('Information content per position', fontdict=font)
ax.set_xticklabels(['HOCOMOCO motifs', 'MuSeAM motifs'], fontdict=font)
#plt.savefig('Entropy_per_position.pdf', format='pdf')
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color='black')
        plt.setp(bp2[element], color='black')
plt.savefig('ICP.pdf')
#plt.show()
