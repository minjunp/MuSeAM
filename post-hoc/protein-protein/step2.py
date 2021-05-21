import sys
import re
import os
import numpy as np
import keras
import tensorflow as tf

print(tf.executing_eagerly())

# Import maxpool outputs
reconstructed_model = keras.models.load_model("../../saved_model/MuSeAM_regression_synthetic_removed/regression_model", compile=False)
maxpool_outs = reconstructed_model.layers[-2].output
print(maxpool_outs)
#print(reconstructed_model.layers[-1].get_weights())
#print(maxpool_outs.shape)
sys.exit()

def maxpool_index_ver2(dir = 'maxpool_index_control_removed'):
    names = []
    for filename in os.listdir(dir):
        if filename.endswith(".txt"):
             name = os.path.join(dir, filename)
             names.append(name)
             continue
        else:
            continue

    nums = []
    for i in range(len(names)):
        f = names[i]

        start = f.find('seq_',0,len(f))
        end = f.find('.txt',0,len(f))
        num = f[start+4:end]
        num = int(num)
        nums.append(num)

    # zip file to sort the dataset
    zipped = zip(nums, names)
    zipped = sorted(zipped)

    nums, files = zip(*zipped)

    for i in range(512):
        filter_index = []
        for j in range(len(files)):
            f = np.loadtxt(files[j])
            filter_index.append(f[i])
        #print(filter_index)
        np.savetxt(os.path.join('./maxpool_index_ver2_control_removed/filter_%d'%i+'.txt'), filter_index)
