
import os
import sys
import numpy as np

def save_model(self, model, alpha, path):
    os.makedirs(f'{path}/motif_original')
    os.makedirs(f'{path}/motif_files')
    model.save(f'{path}/dnase_model')
    #model.save_weights(f'{path}/checkpoint')

    motif_weight = model.get_weights()
    motif_weight = np.asarray(motif_weight[0])

    for i in range(self.filters):
        np.savetxt(f'{path}/motif_original/filter_num_{"{0:0=3d}".format(i)}.txt', motif_weight[:,:,i])

    for i in range(self.filters):
        x = motif_weight[:,:,i]
        berd = np.divide(np.exp(alpha*x), np.transpose(np.expand_dims(np.sum(np.exp(alpha*x), axis = 1), axis = 0), [1,0]))
        np.savetxt(f'{path}/motif_files/filter_num_{"{0:0=3d}".format(i)}.txt', berd)
