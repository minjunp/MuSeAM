import math
import numpy as np
from scipy import stats
import sys

def compute_entropy(motif_matrix):

	motif_matrix = np.array(motif_matrix)
	sum_m = np.sum(motif_matrix, axis = 1)
	normalized_motif = np.divide(motif_matrix, sum_m[:, None])
	motif_matrix = normalized_motif

	r, c = motif_matrix.shape
	assert c == 4

	bkg = [0.295, 0.205, 0.295, 0.205]
	sum_H = 0.

	for i in range(r):
		H = stats.entropy(motif_matrix[i, :], bkg, 2)
		sum_H += H

	return sum_H

# m = [[2, 3, 4, 1], [25, 35, 20, 20], [0.1, 0.4, 0.25, 0.25], [0.8, 0.05, 0.05, 0.1], [0.1, 0.2, 0.3, 0.4], [0.5, 0.3, 0.1, 0.1]]
# entropy = compute_entropy(m)
