import os
import sys
import numpy as np

def save_model(self, model, alpha, path):
    os.makedirs(f'{path}/motif_original')
    os.makedirs(f'{path}/motif_files')
    model.save(f'{path}/model')

    motif_weight = model.get_weights()
    dense_weight = motif_weight[2]
    np.savetxt(f'{path}/dense_weights.txt', dense_weight)

    motifs = np.asarray(motif_weight[0])

    for i in range(self.filters):
        np.savetxt(f'{path}/motif_original/filter_num_{"{0:0=3d}".format(i)}.txt', motifs[:,:,i])

    for i in range(self.filters):
        x = motifs[:,:,i]
        berd = np.divide(np.exp(alpha*x), np.transpose(np.expand_dims(np.sum(np.exp(alpha*x), axis = 1), axis = 0), [1,0]))
        np.savetxt(f'{path}/motif_files/filter_num_{"{0:0=3d}".format(i)}.txt', berd)
