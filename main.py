import models.MuSeAM_classification as MuSeAM_classification
import models.MuSeAM_regression as MuSeAM_regression

from preprocess.split_data import splitData

import numpy as np
import sys

from scipy.stats import spearmanr, pearsonr

import keras
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from keras.models import Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, BatchNormalization, Activation, concatenate, ReLU, Add
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow import keras

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.utils import shuffle

#Reproducibility
seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

class nn_model:
    def __init__(self, fasta_file, readout_file, filters, kernel_size, epochs, batch_size):
        """initialize basic parameters"""
        self.fasta_file = fasta_file
        self.readout_file = readout_file
        self.filters = filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size

        #self.eval()
        self.cross_val()

    def eval(self):
        fwd_train, fwd_test, rc_train, rc_test, readout_train, readout_test = splitData(self.fasta_file,
                                                                                        self.readout_file,
                                                                                        'leaveOneOut',
                                                                                        'binary_classification')

        model = MuSeAM_classification.create_model(self)
        model.fit({'forward': fwd_train, 'reverse': rc_train}, readout_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
        pred_train = model.predict({'forward': fwd_train, 'reverse': rc_train})
        trainAUC = sklearn.metrics.roc_auc_score(readout_train, pred_train)
        print('Train-data AUC is ', trainAUC)

        pred_test = model.predict({'forward': fwd_test, 'reverse': rc_test})
        testAUC = sklearn.metrics.roc_auc_score(readout_test, pred_test)
        print('Test-data AUC is ', testAUC)

    def cross_val(self):
        fwd_fasta, rc_fasta, readout = splitData(self.fasta_file,
                                                self.readout_file,
                                                '10Fold',
                                                'binary_classification')

        # initialize metrics to save values
        trainAUCs = []
        testAUCs = []

        kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        ln = np.zeros(len(readout))

        for train, test in kFold.split(ln, ln):
            model = None
            model = MuSeAM_classification.create_model(self)

            fwdTrain = fwd_fasta[train]
            fwdTest = fwd_fasta[test]
            rcTrain = rc_fasta[train]
            rcTest = rc_fasta[test]
            readoutTrain = readout[train]
            readoutTest = readout[test]

            # Early stopping
            #callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
            model.fit({'forward': fwdTrain, 'reverse': rcTrain}, readoutTrain, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)

            pred_train = model.predict({'forward': fwdTrain, 'reverse': rcTrain})
            trainAUC = sklearn.metrics.roc_auc_score(readoutTrain, pred_train)
            trainAUCs.append(trainAUC)

            pred_test = model.predict({'forward': fwdTest, 'reverse': rcTest})
            testAUC = sklearn.metrics.roc_auc_score(readoutTest, pred_test)
            testAUCs.append(testAUC)

        print(f'Seed number is {seed}')

        print(f'Train 10-fold AUCs are {trainAUCs}')
        print(f'Mean train AUC is {np.mean(trainAUCs)}')

        print(f'Test 10-fold AUCs are {testAUCs}')
        print(f'Mean test AUC is {np.mean(testAUCs)}')
