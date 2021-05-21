import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

#plt.style.use('ggplot')
font = {'family' : 'Arial',
        #'weight' : 'bold',
        'size': 10}

plt.rc('font', **font)

df = pd.read_csv('html_input.txt', sep=' ', header=None)
motifs = df[2].values
dense_weight = df[5].values

motifs = np.concatenate((motifs[0:10], motifs[-10:]))
energies = np.concatenate((dense_weight[0:10], dense_weight[-10:]))
print(motifs)
print(energies)

# x1 = ['FOXP1', 'BHA15', 'FOXJ2', 'TFAP4', 'NR4A1', 'SOX17', 'PBX3', 'ZN335', 'RFX5', 'NR4A2']
# energy1 = [-0.1924471, -0.19227242, -0.17712532, -0.16648611, -0.15690519, -0.15640533, -0.15459466, -0.14704163, -0.1436076, -0.14298865]
# x2 = ['FOSB', 'NF2L2', 'FOSL2', 'ZNF41', 'JUN', 'CLOCK', 'BACH2', 'ETS1', 'JUND', 'ATF1']
# energy2 = [0.212049857, 0.19189301, 0.18950242, 0.17583951, 0.17286016, 0.17248528, 0.17122647, 0.16984972, 0.16289043, 0.1602487713]
# energy2 = sorted(energy2)

positions = np.arange(20)
width = 0.35
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
x = np.arange(len(motifs))

ax.bar(x, energies, color='#d95f0e')
ax2.bar(x, energies, color='#d95f0e')

ax.set_ylim(0.2, 0.4)  # outliers only
ax2.set_ylim(0, -0.001)  # most of the data
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
