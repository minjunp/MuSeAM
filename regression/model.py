from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import sys
import math
import os
import json
import csv
import pandas
import keras

from keras.utils.vis_utils import model_to_dot

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, BatchNormalization, Activation, concatenate, ReLU, Dropout

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr

import matplotlib.pyplot as plt

from data_preprocess import preprocess
from sklearn.utils import shuffle
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow import keras
from keras.models import Model
from numpy import newaxis

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ConvolutionLayer(Conv1D):
    def __init__(self, filters,
                 kernel_size,
                 data_format,
                 padding='valid',
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 __name__ = 'ConvolutionLayer',
                 **kwargs):
        super(ConvolutionLayer, self).__init__(filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            **kwargs)
        self.run_value = 1

    def call(self, inputs):

      ## shape of self.kernel is (12, 4, 512)
      ##the type of self.kernel is <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>
        if self.run_value > 2:

            x_tf = self.kernel  ##x_tf after reshaping is a tensor and not a weight variable :(
            x_tf = tf.transpose(x_tf, [2, 0, 1])

            alpha = 100
            beta = 1/alpha
            bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
            bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(lambda x:
                                  tf.math.scalar_mul(beta, tf.subtract(tf.subtract(tf.subtract(tf.math.scalar_mul(alpha, x),
                                  tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis = 1), axis = 1)),
                                  tf.expand_dims(tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.subtract(tf.math.scalar_mul(alpha, x),
                                  tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis = 1), axis = 1))), axis = 1)), axis = 1)),
                                  tf.math.log(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]])))), x_tf)
            #print("type of output from map_fn is", type(filt_list)) ##type of output from map_fn is <class 'tensorflow.python.framework.ops.Tensor'>   shape of output from map_fn is (10, 12, 4)
            #print("shape of output from map_fn is", filt_list.shape)
            #transf = tf.reshape(filt_list, [12, 4, self.filters]) ##12, 4, 512
            transf = tf.transpose(filt_list, [1, 2, 0])
            ##type of transf is <class 'tensorflow.python.framework.ops.Tensor'>
            outputs = self._convolution_op(inputs, transf) ## type of outputs is <class 'tensorflow.python.framework.ops.Tensor'>

        else:
            outputs = self._convolution_op(inputs, self.kernel)
        self.run_value += 1
        return outputs

class nn_model:
    def __init__(self, fasta_file, readout_file, filters, kernel_size, pool_type, regularizer, activation_type, epochs, batch_size, loss_func, optimizer):
        """initialize basic parameters"""
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_type = pool_type
        self.regularizer = regularizer
        self.activation_type = activation_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.fasta_file = fasta_file
        self.readout_file = readout_file
        self.loss_func = loss_func
        self.optimizer = optimizer

        #self.eval()
        self.cross_val()
        #self.cross_val_binning()

    def create_model(self):
        # different metric functions
        def coeff_determination(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res/(SS_tot + K.epsilon()))
        def spearman_fn(y_true, y_pred):
            return tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                   tf.cast(y_true, tf.float32)], Tout=tf.float32)
        #Create new loss function (Rank mse)
        @tf.function()
        def rank_mse(yTrue, yPred):
          lambda_value=0.25
          #pass lambda value as tensor
          lambda_value = tf.convert_to_tensor(lambda_value,dtype="float32")

          #get vector ranks
          rank_yTrue = tf.argsort(tf.argsort(yTrue))
          rank_yPred = tf.argsort(tf.argsort(yPred))

          #calculate losses
          mse = tf.reduce_mean(tf.square(tf.subtract(yTrue,yPred)))
          rank_mse = tf.reduce_mean(tf.square(tf.subtract(rank_yTrue,rank_yPred)))

          #take everything to same dtype
          mse = tf.cast(mse,dtype="float32")
          rank_mse = tf.cast(rank_mse,dtype="float32")

          #(1 - lambda value)* mse(part a of loss)
          loss_a = tf.multiply(tf.subtract(tf.ones(1,dtype="float32"),lambda_value),mse)
          #lambda value * rank_mse (part b of loss)
          loss_b = tf.multiply(lambda_value,rank_mse)
          #final loss
          loss = tf.add(loss_a,loss_b)

          return loss

        # building model
        prep = preprocess(self.fasta_file, self.readout_file)
        # if want mono-nucleotide sequences
        dict = prep.one_hot_encode()
        # if want dinucleotide sequences
        #dict = prep.dinucleotide_encode()

        readout = dict["readout"]
        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]

        dim_num = fw_fasta.shape

        # To build this model with the functional API,
        # you would start by creating an input node:
        forward = keras.Input(shape=(dim_num[1],dim_num[2]), name = 'forward')
        reverse = keras.Input(shape=(dim_num[1],dim_num[2]), name = 'reverse')

        #first_layer = Conv1D(filters=300, kernel_size=19, data_format='channels_last', input_shape=(dim_num[1],dim_num[2]), use_bias = True)
        first_layer = Conv1D(filters=300, kernel_size=19)

        #first_layer = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, strides=1, data_format='channels_last', use_bias = True)

        conv1_fwd = first_layer(forward)
        conv1_rc = first_layer(reverse)

        concat = concatenate([conv1_fwd, conv1_rc], axis=1)

        batch_norm1 = BatchNormalization()(concat)

        relu = ReLU()(batch_norm1)

        max1 = MaxPooling1D(pool_size=3)(relu)

        conv2 = Conv1D(filters=200, kernel_size=11)(max1)

        batch_norm2 = BatchNormalization()(conv2)

        relu2 = ReLU()(batch_norm2)

        max2 = MaxPooling1D(pool_size=4)(relu2)

        conv3 = Conv1D(filters=200, kernel_size=7)(max2)

        batch_norm3 = BatchNormalization()(conv3)

        relu3 = ReLU()(batch_norm3)

        max3 = MaxPooling1D(pool_size=4)(relu3)

        flat = Flatten()(max3)

        linear1 = Dense(1000)(flat)

        relu4 = ReLU()(linear1)

        dropout1 = Dropout(rate=0.3)(relu4)

        linear2 = Dense(1000)(dropout1)

        relu5 = ReLU()(linear2)

        dropout2 = Dropout(rate=0.3)(relu5)

        linear3 = Dense(164)(dropout2)

        #sigmoid_out = tf.keras.activations.sigmoid(linear3)
        #sigmoid_out = Dense(1, activation='sigmoid')(linear3)
        outputs = Dense(1)(linear3)

        model = keras.Model(inputs=[forward, reverse], outputs=outputs)

        model.summary()

        if self.loss_func == 'mse':
            model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics = coeff_determination)
        elif self.loss_func == 'huber':
            loss_huber = keras.losses.Huber(delta=1)
            model.compile(loss=loss_huber, optimizer=self.optimizer, metrics = coeff_determination)
        elif self.loss_func == 'mae':
            loss_mae = keras.losses.MeanAbsoluteError()
            model.compile(loss=loss_mae, optimizer=self.optimizer, metrics = coeff_determination)
        elif self.loss_func == 'rank_mse':
            model.compile(loss=rank_mse, optimizer=self.optimizer, metrics = coeff_determination)
        else:
            raise NameError('Unrecognized Loss Function')

        return model

    def eval(self):

        # Preprocess the data to one-hot encoded vector
        prep = preprocess(self.fasta_file, self.readout_file)

        dict = prep.one_hot_encode()

        # if want dinucleotide sequences
        # dict = prep.dinucleotide_encode()

        # print maximum length without truncation
        np.set_printoptions(threshold=sys.maxsize)

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        if self.activation_type == 'linear':
            readout = np.log2(readout)
            readout = np.ndarray.tolist(readout)

        seed = random.randint(1,1000)
        #seed = 460

        # 90% Train, 10% Test
        x1_train, x1_test, y1_train, y1_test = train_test_split(fw_fasta, readout, test_size=0.1, random_state=seed)
        x2_train, x2_test, y2_train, y2_test = train_test_split(rc_fasta, readout, test_size=0.1, random_state=seed)

        model = self.create_model()

        # change from list to numpy array
        y1_train = np.asarray(y1_train)
        y1_test = np.asarray(y1_test)
        y2_train = np.asarray(y2_train)
        y2_test = np.asarray(y2_test)

        # Without early stopping
        history = model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)

        # Early stopping
        #callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        #callback = EarlyStopping(monitor='val_spearman_fn', min_delta=0.0001, patience=3, verbose=0, mode='max', baseline=None, restore_best_weights=False)
        #history = model.fit({'forward': x1_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks = [callback])

        history2 = model.evaluate({'forward': x1_test, 'reverse': x2_test}, y1_test)
        pred = model.predict({'forward': x1_test, 'reverse': x2_test})

        pred = np.reshape(pred, len(pred))
        pred = pred.tolist()
        y2_test = y2_test.tolist()
        spearman = spearmanr(y2_test, pred)
        #viz_prediction(pred, y1_test, '{} regression model'.format(self.loss_func), '{}2.png'.format(self.loss_func))

        print('spearman is ', spearman)
        print("Seed number is {}".format(seed))
        print('metric values of model.evaluate: '+ str(history2))
        print('metrics names are ' + str(model.metrics_names))

    def cross_val(self):

        # Preprocess the data
        prep = preprocess(self.fasta_file, self.readout_file)
        dict = prep.one_hot_encode()
        # If want dinucleotide sequences
        #dict = prep.dinucleotide_encode()

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        if self.activation_type == 'linear':
            readout = np.log2(readout)
            readout = np.ndarray.tolist(readout)

        # seed to reproduce results
        seed = random.randint(1,1000)
        seed = 460

        forward_shuffle, readout_shuffle = shuffle(fw_fasta, readout, random_state=seed)
        reverse_shuffle, readout_shuffle = shuffle(rc_fasta, readout, random_state=seed)
        readout_shuffle = np.array(readout_shuffle)

        # initialize metrics to save values
        metrics = []
        spearmans = []

        # Provides train/test indices to split data in train/test sets.
        kFold = StratifiedKFold(n_splits=10)
        ln = np.zeros(len(readout_shuffle))
        true_vals = []
        pred_vals = []

        for train, test in kFold.split(ln, ln):
            model = None
            model = self.create_model()

            fwd_train = forward_shuffle[train]
            fwd_test = forward_shuffle[test]
            rc_train = reverse_shuffle[train]
            rc_test = reverse_shuffle[test]
            y_train = readout_shuffle[train]
            y_test = readout_shuffle[test]

            # Early stopping
            #callback = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, verbose=0, mode='max', baseline=None, restore_best_weights=False)
            #callback = EarlyStopping(monitor='val_spearman_fn', min_delta=0.0001, patience=3, verbose=0, mode='max', baseline=None, restore_best_weights=False)
            #history = model.fit({'forward': fwd_train, 'reverse': rc_train}, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0, callbacks = [callback])

            #history = model.fit({'forward': fwd_train, 'reverse': rc_train}, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)
            history = model.fit({'forward': fwd_train, 'reverse': rc_train}, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)
            history2 = model.evaluate({'forward': fwd_test, 'reverse': rc_test}, y_test)
            pred = model.predict({'forward': fwd_test, 'reverse': rc_test})

            pred = np.reshape(pred, len(pred))
            pred = pred.tolist()
            y_test = y_test.tolist()
            spearman = spearmanr(y_test, pred)[0]
            #viz_prediction(pred, y1_test, '{} regression model'.format(self.loss_func), '{}2.png'.format(self.loss_func))

            print('spearman is ', spearman)
            metrics.append(history2)
            spearmans.append(spearman)
            pred = np.reshape(pred,len(pred))
            true_vals.append(y_test)
            pred_vals.append(pred)

        g1 = []
        g2 = []
        for i in metrics:
            loss, r_2 = i
            g1.append(loss)
            g2.append(r_2)

        #np.savetxt('true_vals.txt', true_vals)
        #np.savetxt('pred_vals.txt', pred_vals)
        #viz_prediction(pred_vals, true_vals, '{} delta=1 regression model (seed=460)'.format(self.loss_func), '{}_d1.png'.format(self.loss_func))

        print(g2)
        print(spearmans)
        print('seed number = %d' %seed)
        print('Mean loss of 10-fold cv is ' + str(np.mean(g1)))
        print('Mean R_2 score of 10-fold cv is ' + str(np.mean(g2)))
        print('Mean Spearman of 10-fold cv is ' + str(np.mean(spearmans)))

    def cross_val_binning(self):
        # Preprocess the data
        prep = preprocess(self.fasta_file, self.readout_file)
        dict = prep.one_hot_encode()
        # If want dinucleotide sequences
        #dict = prep.dinucleotide_encode()

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        if self.activation_type == 'linear':
            readout = np.log2(readout)

        # Returns the indices that would sort an array.
        x = np.argsort(readout)
        """
        ind = []
        num = 10
        for i in range(num):
            for j in range(int(len(x)/num)):
                id = i + j * num
                ind.append(x[id])

        forward_bin = fw_fasta[ind]
        reverse_bin = rc_fasta[ind]
        readout_bin = readout[ind]
        """
        forward_bin = fw_fasta[x]
        reverse_bin = rc_fasta[x]
        readout_bin = readout[x]

        # initialize metrics to save values
        metrics = []

        # Provides train/test indices to split data in train/test sets.
        kFold = StratifiedKFold(n_splits=10)
        ln = np.zeros(len(readout_bin))
        true_vals = []
        pred_vals = []

        for train, test in kFold.split(ln, ln):
            model = None
            model = self.create_model()

            fwd_train = forward_bin[train]
            fwd_test = forward_bin[test]
            rc_train = reverse_bin[train]
            rc_test = reverse_bin[test]
            y_train = readout_bin[train]
            y_test = readout_bin[test]

            # Early stopping
            #callback = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, verbose=0, mode='max', baseline=None, restore_best_weights=False)
            #callback = EarlyStopping(monitor='val_spearman_fn', min_delta=0.0001, patience=3, verbose=0, mode='max', baseline=None, restore_best_weights=False)
            #history = model.fit({'forward': fwd_train, 'reverse': rc_train}, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0, callbacks = [callback])

            history = model.fit({'forward': fwd_train, 'reverse': rc_train}, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)
            history2 = model.evaluate({'forward': fwd_test, 'reverse': rc_test}, y_test)
            pred = model.predict({'forward': fwd_test, 'reverse': rc_test})

            metrics.append(history2)
            pred = np.reshape(pred,len(pred))
            true_vals.append(y_test.tolist())
            pred_vals.append(pred.tolist())

        g1 = []
        g2 = []
        g3 = []
        for i in metrics:
            loss, r_2, spearman_val = i
            g1.append(loss)
            g2.append(r_2)
            g3.append(spearman_val)

        #np.savetxt('true_vals.txt', true_vals)
        #np.savetxt('pred_vals.txt', pred_vals)
        viz_prediction(pred_vals, true_vals, '{} regression model'.format(self.loss_func), '{}_binning_ascending.png'.format(self.loss_func))

        print(g2)
        print(g3)
        print('Mean loss of 10-fold cv is ' + str(np.mean(g1)))
        print('Mean R_2 score of 10-fold cv is ' + str(np.mean(g2)))
        print('Mean Spearman of 10-fold cv is ' + str(np.mean(g3)))

def viz_prediction(pred, true, title_name, file_name):
    # plot true vs pred
    plt.plot(pred, true, 'o', markersize=1)
    plt.xlabel('Predicted Enhancer Activity')
    plt.ylabel('True Enhancer Activity')
    plt.title(title_name)
    plt.xlim(-0.75, 3)
    plt.ylim(-0.75, 3)
    plt.savefig(file_name)
