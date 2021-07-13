import pandas as pd
import numpy as np
import sys

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

shuffled_vals = np.loadtxt('shuffled.txt')
liver_ehancer_vals = np.loadtxt('liver_enhancer.txt')

# data = [data, d2, d2[::2,0]]
data = [shuffled_vals, liver_ehancer_vals]
fig, ax = plt.subplots()
ax.set_title('Box Plot')
ax.boxplot(data)

pval_less = stats.ks_2samp(shuffled_vals, liver_ehancer_vals, alternative='less')
print('ks 2sample with one-tailed (alternative = less) is:')
print(pval_less)

plt.show()
