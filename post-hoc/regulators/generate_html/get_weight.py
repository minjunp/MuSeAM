import sys
import keras
import numpy as np

"""
# Reconstruct model from the saved model
reconstructed_model = keras.models.load_model("../../saved_model/MuSeAM_regression/regression_model", compile=False)
model_weights = reconstructed_model.get_weights()

dense_weight = model_weights[2]
dense_bias = model_weights[3]
"""

# dense_weight = np.loadtxt('/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_liver_enhancer/dense_weights.txt')
dense_weight = np.loadtxt('/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_silencer/dense_weights.txt')

dense_weight = np.reshape(dense_weight, (512,))
dense_weight_sorted = np.argsort(dense_weight)

# np.savetxt('./liver_enhancer/motif_rank.txt', dense_weight_sorted)
np.savetxt('./silencer/motif_rank.txt', dense_weight_sorted)

# --------------------------------------------------------------------
## Top regulators (From low to high values)
activator_indices = dense_weight_sorted[-10:]
repressor_indices = dense_weight_sorted[0:10]
# print(dense_weight[activator_indices])
# print(dense_weight[repressor_indices])
# print(activator_indices)
# print(repressor_indices)
