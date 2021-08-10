import tensorflow as tf
import sys
import keras
import numpy as np

"""
reconstructed_model = keras.models.load_model("../../saved_model/MuSeAM_regression/regression_model", compile=False)
model_weights = reconstructed_model.get_weights()

dense_weight = model_weights[2]
dense_bias = model_weights[3]
"""

dense_weight = np.loadtxt('./silencer/dense_weights_silencer.txt')
dense_weight = np.reshape(dense_weight, (512,))
## returns indices in ascending order
dense_weight_sorted = np.argsort(dense_weight)

## Top regulators (From low to high values)
activator_indices = dense_weight_sorted[-10:]
repressor_indices = dense_weight_sorted[0:10]

print(dense_weight[activator_indices])
print(dense_weight[repressor_indices])
print(activator_indices)
print(repressor_indices)

# [0.2535086  0.25505617 0.2573022  0.2670092  0.27770907 0.28143048
#  0.28210825 0.28237686 0.328803   0.33938998]
# [-0.00164494 -0.00041562 -0.00029084 -0.00026359 -0.00025761 -0.00022864
#  -0.00022482 -0.00022109 -0.00020093 -0.00017169]
# [359  75 411 210 276 347  73 231 361 241]
# [274 372  55 170  63  67 265 173 236 301]

np.savetxt('motif_silencer_rank.txt', dense_weight_sorted)
#np.savetxt('dense_weights.txt', dense_weight)
#np.savetxt('dense_bias.txt', dense_bias)
