import models.MuSeAM_classification as MuSeAM_classification
import models.MuSeAM_regression as MuSeAM_regression
import models.MuSeAM_sharpr as MuSeAM_sharpr
import models.MuSeAM_averagePooling as MuSeAM_averagePooling
import models.sharpr as sharpr_model
import models.MuSeAM_sharpr_single_input as MuSeAM_sharpr_single_input
import models.MuSeAM_horizontal as MuSeAM_horizontal
import models.MuSeAM_skip_connection as MuSeAM_skip_connection


from saved_model import save_model

from preprocess.split_data import splitData, sharpr
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
seed = 7163
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
        #self.cross_val()
        self.eval_sharpr()
        #self.load_weights()
        #self.fitAll()

    def eval_sharpr(self):
        #dtype = 'two-inputs'
        dtype = 'one-input'

        if dtype == 'two-inputs':
            fwd_train, rc_train, readout_train, fwd_test, rc_test, readout_test, fwd_valid, rc_valid, readout_valid = sharpr(dtype)
            #model = MuSeAM_sharpr.create_model(self)
            #model = MuSeAM_skip_connection.create_model(self)
            #model = MuSeAM_horizontal.create_model(self)
            model = MuSeAM_averagePooling.create_model(self)
            #model = sharpr_model.create_model(self)

            model.fit({'forward': fwd_train, 'reverse': rc_train}, readout_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=({'forward': fwd_valid, 'reverse': rc_valid}, readout_valid))
            predictions = model.predict({'forward': fwd_test, 'reverse': rc_test})
            predictions_valid = model.predict({'forward': fwd_valid, 'reverse': rc_valid})

        if dtype == 'one-input':
            train_seq, train_readout, fwd_test, readout_test, fwd_valid, readout_valid = sharpr(dtype)
            model = MuSeAM_sharpr_single_input.create_model(self)

            model.fit(train_seq, train_readout, epochs=self.epochs, batch_size=self.batch_size, validation_data=(fwd_valid, readout_valid))
            predictions = model.predict(fwd_test)
            predictions_valid = model.predict(fwd_valid)

        # Loop over each task
        spearmans = [spearmanr(readout_test[:, i], predictions[:, i])[0] for i in range(12)]
        pearsons = [pearsonr(readout_test[:, i], predictions[:, i])[0] for i in range(12)]

        spearmans_valid = [spearmanr(readout_valid[:, i], predictions_valid[:, i])[0] for i in range(12)]
        pearsons_valid = [pearsonr(readout_valid[:, i], predictions_valid[:, i])[0] for i in range(12)]

        print(f'Test spearman averages are: {np.array(spearmans)[[2,5,8,11]]}')
        print(f'Test Mean spearman is {np.mean(spearmans)}')

        print(f'Validation spearman averages are: {np.array(spearmans_valid)[[2,5,8,11]]}')
        print(f'Validation Mean spearman is {np.mean(spearmans_valid)}')

    def load_weights(self):
        reconstructed_model = keras.models.load_model("./saved_model/MuSeAM_regression/regression_model", compile=False)

        #dtype = 'two-inputs'
        dtype = 'one-input'

        if dtype == 'two-inputs':
            fwd_train, rc_train, readout_train, fwd_test, rc_test, readout_test,fwd_valid, rc_valid, readout_valid = sharpr()
            model = MuSeAM_sharpr.create_model(self)
            #print(reconstructed_model.layers[2].get_weights()[0].shape)
            filters = reconstructed_model.layers[2].get_weights()
            model.layers[2].set_weights(filters)

            model.fit({'forward': fwd_train, 'reverse': rc_train}, readout_train, epochs=self.epochs, batch_size=self.batch_size)
            predictions = model.predict({'forward': fwd_test, 'reverse': rc_test})
        if dtype == 'one-input':
            train_seq, train_readout, fwd_test, readout_test, fwd_valid, readout_valid = sharpr(dtype)
            model = MuSeAM_sharpr_single_input.create_model(self)
            filters = reconstructed_model.layers[2].get_weights()
            model.layers[1].set_weights(filters)

            model.fit(train_seq, train_readout, epochs=self.epochs, batch_size=self.batch_size, validation_data=(fwd_valid, readout_valid))
            predictions = model.predict(fwd_test)

        # Loop over each task
        spearmans = [spearmanr(readout_test[:, i], predictions[:, i])[0] for i in range(12)]
        pearsons = [pearsonr(readout_test[:, i], predictions[:, i])[0] for i in range(12)]

        #print(f'pearson averages are: {np.array(pearsons)[[2,5,8,11]]}')
        print(f'spearman averages are: {np.array(spearmans)[[2,5,8,11]]}')
        print(f'Mean spearman is {np.mean(spearmans)}')


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

    def fitAll(self):
        fwd_fasta, rc_fasta, readout = splitData(self.fasta_file,
                                                self.readout_file,
                                                'fitAll')
        model = MuSeAM_regression.create_model(self)
        model.fit({'forward': fwd_fasta, 'reverse': rc_fasta}, readout, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)
        history = model.evaluate({'forward': fwd_fasta, 'reverse': rc_fasta}, readout)
        print(history) # [0.045967694371938705, 0.931830644607544, 0.9492059946060181]

        save_model.save_model(self, model, alpha=120, path='./saved_model/MuSeAM_regression')

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
