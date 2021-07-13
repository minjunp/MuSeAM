import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

#plt.style.use('ggplot')
font = {'family' : 'Arial',
        #'weight' : 'bold',
        'size': 10}

plt.rc('font', **font)

# html file path
filePath = '/Users/minjunp/Documents/baylor/MuSeAM/post-hoc/regulators/generate_html/liver_enhancer/html_input_evalue_10.txt'
df = pd.read_csv(filePath, sep=' ', header=None)
TFs = df[2].values
dense_weight = df[5].values

motifs = TFs[0:20]
energies = dense_weight[0:20]

positions = np.arange(20)
width = 0.35
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
x = np.arange(len(motifs))

ax.bar(x, energies, color='#d95f0e')
ax2.bar(x, energies, color='#d95f0e')

ax.set_ylim(0, 0.4)  # outliers only
#ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax.set_facecolor('xkcd:white')
ax2.set_facecolor('xkcd:white')

ax.grid(linewidth='0.2', color='grey', axis='y')
ax2.grid(linewidth='0.2', color='grey', axis='y')
#ax.grid(True)
ax2.set_xticks(x)
ax2.set_xticklabels(motifs)
ax2.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right', fontsize=8)
plt.show()
#plt.savefig('act_rep.pdf')
