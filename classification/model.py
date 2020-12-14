from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
import sys
import math
import os
import json
import csv
import pandas
import sklearn
import pandas as pd


# specifically for model visualization
import keras
#import pydot as pyd

from keras.utils.vis_utils import model_to_dot

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, BatchNormalization, Activation, concatenate, ReLU
#import keras
#from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import tensorflow as tf
#from tensorflow.nn import avg_pool1d
from scipy.stats import spearmanr, pearsonr
import scipy

import matplotlib.pyplot as plt

from data_preprocess import preprocess
from sklearn.utils import shuffle
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow import keras
from keras.models import Model
from numpy import newaxis

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

        print("self.run value is", self.run_value)
        if self.run_value > 2:

            x_tf = self.kernel  ##x_tf after reshaping is a tensor and not a weight variable :(
            x_tf = tf.transpose(x_tf, [2, 0, 1])

            alpha = 1000
            beta = 1/alpha
            bkg = tf.constant([0.25, 0.25, 0.25, 0.25])
            bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(lambda x: tf.math.scalar_mul(beta, tf.subtract(tf.subtract(tf.subtract(tf.math.scalar_mul(alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis = 1), axis = 1)), tf.expand_dims(tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.subtract(tf.math.scalar_mul(alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis = 1), axis = 1))), axis = 1)), axis = 1)), tf.math.log(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]])))), x_tf)
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
    def __init__(self, fasta_file, readout_file, filters, kernel_size, pool_type, pool, regularizer, activation_type, epochs, batch_size):
        """initialize basic parameters"""
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_type = pool_type
        self.pool = pool
        self.regularizer = regularizer
        self.activation_type = activation_type
        self.epochs = epochs
        self.batch_size = batch_size
        #self.stride = stride
        #self.create_model()
        self.fasta_file = fasta_file
        self.readout_file = readout_file

        self.eval()
        #self.filter_importance()
        #self.cross_val_custom()

    def create_model(self):
        # different metric functions
        def coeff_determination(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res/(SS_tot + K.epsilon()))


        def auroc(y_true, y_pred):
            return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

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

        #first_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', input_shape=(dim_num[1],dim_num[2]), use_bias = False)
        ## with trainable = False
        #first_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, kernel_initializer = my_init, data_format='channels_last', input_shape=(dim_num[1],dim_num[2]), use_bias = False, trainable=False)
        first_layer = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', use_bias = True)


        fw = first_layer(forward)
        bw = first_layer(reverse)

        concat_relu = concatenate([fw, bw], axis=1)

        #concat_relu = ReLU()(concat)
        #concat_relu = Dense(1, activation= 'sigmoid')(concat)

        if self.pool_type == 'Max':
            pool_layer = MaxPooling1D(pool_size=self.pool)(concat_relu)
        elif self.pool_type == 'Ave':
            pool_layer = AveragePooling1D(pool_size=self.pool)(concat_relu)
        elif self.pool_type == 'custom':

            def out_shape(input_shape):
                shape = list(input_shape)
                print(input_shape)
                shape[0] = 10
                return tuple(shape)
            #model.add(Lambda(top_k, arguments={'k': 10}))

            def top_k(inputs, k):
                # tf.nn.top_k Finds values and indices of the k largest entries for the last dimension
                print(inputs.shape)
                inputs2 = tf.transpose(inputs, [0,2,1])
                new_vals = tf.nn.top_k(inputs2, k=k, sorted=True).values
                # transform back to (None, 10, 512)
                return tf.transpose(new_vals, [0,2,1])

            pool_layer = Lambda(top_k, arguments={'k': 2})(concat_relu)
            pool_layer = AveragePooling1D(pool_size=2)(pool_layer)
        elif self.pool_type == 'custom_sum':
            ## apply relu function before custom_sum functions

            def summed_up(inputs):
                #nonzero_vals = tf.keras.backend.relu(inputs)
                new_vals = tf.math.reduce_sum(inputs, axis = 1, keepdims = True)
                return new_vals
            pool_layer = Lambda(summed_up)(concat_relu)

        else:
            sys.exit()

        # flatten the layer (None, 512)
        flat = Flatten()(pool_layer)

        after_flat = Dense(32)(flat)

        if self.regularizer == 'L_1':
            #outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(flat)
            ## trainable = False with learned bias

            #outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(after_flat)
            outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= 'sigmoid')(after_flat)
        elif self.regularizer == 'L_2':
            #outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(flat)
            ## trainable = False with learned bias
            outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation= self.activation_type)(after_flat)
        else:
            sys.exit()

        #weight_forwardin_0=model.layers[0].get_weights()[0]
        #print(weight_forwardin_0)
        model = keras.Model(inputs=[forward, reverse], outputs=outputs, name='mpra_model')

        #print model summary
        model.summary()

        #model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])

        return model

    def filter_importance(self):
        prep = preprocess(self.fasta_file, self.readout_file)

        # if want mono-nucleotide sequences
        dict = prep.one_hot_encode()

        # if want dinucleotide sequences
        # dict = prep.dinucleotide_encode()

        # print maximum length without truncation
        np.set_printoptions(threshold=sys.maxsize)

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        seed = random.randint(1,1000)

        x1_train, x1_test, y1_train, y1_test = train_test_split(fw_fasta, readout, test_size=0.1, random_state=seed)
        # split for reverse complemenet sequences
        x2_train, x2_test, y2_train, y2_test = train_test_split(rc_fasta, readout, test_size=0.1, random_state=seed)
        #assert x1_test == x2_test
        #assert y1_test == y2_test

        model = self.create_model()

        # Restore the weights
        train_afters = []
        test_afters = []
        for i in range(self.filters):
            model.load_weights('./my_checkpoint')
            weights = model.get_weights()

            zeros = np.zeros((12,4))
            #weights[0][:,:,i] = zeros

            #model.set_weights(weights)

            pred_train = model.predict({'forward': x1_train, 'reverse': x2_train})
            vals = []
            for i in range(len(pred_train)):
                if pred_train[i] < 0.5:
                    val = 0
                    vals.append(val)
                if pred_train[i] >= 0.5:
                    val = 1
                    vals.append(val)

            print(y1_train[0:10])
            print(vals[0:10])

            true_pred = 0
            false_pred = 0
            for ind in range(len(pred_train)):
                if y1_train[ind] == vals[ind]:
                    true_pred += 1
                else:
                    false_pred += 1
            print('Total number of train-set predictions is: ' + str(len(y1_train)))
            print('Number of correct train-set predictions is: ' + str(true_pred))
            print('Number of incorrect train-set predictions is: ' + str(false_pred))

            auc_score = sklearn.metrics.roc_auc_score(y1_train, pred_train)
            print('train-set auc score is: ' + str(auc_score))
            print('train-set seed number is: ' + str(seed))
            train_afters.append(auc_score)

            ##########################################################

            pred = model.predict({'forward': x1_test, 'reverse': x2_test})

            vals = []
            for i in range(len(pred)):
                if pred[i] < 0.5:
                    val = 0
                    vals.append(val)
                if pred[i] >= 0.5:
                    val = 1
                    vals.append(val)


            true_pred = 0
            false_pred = 0
            for ind in range(len(y1_test)):
                if y1_test[ind] == vals[ind]:
                    true_pred += 1
                else:
                    false_pred += 1
            print('Total number of test-set predictions is: ' + str(len(y1_test)))
            print('Number of correct test-set predictions is: ' + str(true_pred))
            print('Number of incorrect test-set predictions is: ' + str(false_pred))

            auc_score = sklearn.metrics.roc_auc_score(y1_test, pred)
            print('test-set auc score is: ' + str(auc_score))
            print('test-set seed number is: ' + str(seed))
            test_afters.append(auc_score)
        print('***********************************************************************')
        print(train_afters)
        print(test_afters)

    def eval(self):
        prep = preprocess(self.fasta_file, self.readout_file)

        # if want mono-nucleotide sequences
        dict = prep.one_hot_encode()

        # if want dinucleotide sequences
        # dict = prep.dinucleotide_encode()

        # print maximum length without truncation
        np.set_printoptions(threshold=sys.maxsize)

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        seed = random.randint(1,1000)

        x1_train, x1_test, y1_train, y1_test = train_test_split(fw_fasta, readout, test_size=0.1, random_state=seed)
        # split for reverse complemenet sequences
        x2_train, x2_test, y2_train, y2_test = train_test_split(rc_fasta, readout, test_size=0.1, random_state=seed)
        #assert x1_test == x2_test
        #assert y1_test == y2_test

        model = self.create_model()


        # change from list to numpy array
        y1_train = np.asarray(y1_train)
        y1_test = np.asarray(y1_test)
        y2_train = np.asarray(y2_train)
        y2_test = np.asarray(y2_test)

        # if we want to merge two training dataset
        # comb = np.concatenate((y1_train, y2_train))

        # train the data
        model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)

        # Save the entire model as a SavedModel.
        #model.save('my_model')
        # Save weights only
        #model.save_weights('./my_checkpoint')

        pred_train = model.predict({'forward': x1_train, 'reverse': x2_train})
        vals = []
        for i in range(len(pred_train)):
            if pred_train[i] < 0.5:
                val = 0
                vals.append(val)
            if pred_train[i] >= 0.5:
                val = 1
                vals.append(val)

        print(y1_train[0:10])
        print(vals[0:10])

        true_pred = 0
        false_pred = 0
        for ind in range(len(pred_train)):
            if y1_train[ind] == vals[ind]:
                true_pred += 1
            else:
                false_pred += 1
        print('Total number of train-set predictions is: ' + str(len(y1_train)))
        print('Number of correct train-set predictions is: ' + str(true_pred))
        print('Number of incorrect train-set predictions is: ' + str(false_pred))

        auc_score = sklearn.metrics.roc_auc_score(y1_train, pred_train)
        print('train-set auc score is: ' + str(auc_score))
        print('train-set seed number is: ' + str(seed))

        ##########################################################

        pred = model.predict({'forward': x1_test, 'reverse': x2_test})

        vals = []
        for i in range(len(pred)):
            if pred[i] < 0.5:
                val = 0
                vals.append(val)
            if pred[i] >= 0.5:
                val = 1
                vals.append(val)


        true_pred = 0
        false_pred = 0
        for ind in range(len(y1_test)):
            if y1_test[ind] == vals[ind]:
                true_pred += 1
            else:
                false_pred += 1
        print('Total number of test-set predictions is: ' + str(len(y1_test)))
        print('Number of correct test-set predictions is: ' + str(true_pred))
        print('Number of incorrect test-set predictions is: ' + str(false_pred))

        auc_score = sklearn.metrics.roc_auc_score(y1_test, pred)
        print('test-set auc score is: ' + str(auc_score))
        print('test-set seed number is: ' + str(seed))

    def cross_val_custom(self):
        # so that we get different metrics used in this custom version
        # preprocess the data
        prep = preprocess(self.fasta_file, self.readout_file)

        # if want mono-nucleotide sequences
        dict = prep.one_hot_encode()
        # if want dinucleotide sequences
        #dict = prep.dinucleotide_encode()

        np.set_printoptions(threshold=sys.maxsize)

        # seed to reproduce results
        seed = random.randint(1,1000)

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        #if self.activation_type == 'linear':
        #    readout = np.log2(readout)
        #    readout = np.ndarray.tolist(readout)

        step = list(range(0,len(fw_fasta), int(len(fw_fasta)/10)))
        step.append(len(fw_fasta))
        step.remove(0)

        forward_shuffle, readout_shuffle = shuffle(fw_fasta, readout, random_state=seed)
        reverse_shuffle, readout_shuffle = shuffle(rc_fasta, readout, random_state=seed)

        forward_shuffle = np.ndarray.tolist(forward_shuffle)
        reverse_shuffle = np.ndarray.tolist(reverse_shuffle)

        # initialize metrics to save values
        metrics = []
        save_pred = []

        #fig = plt.figure()
        index = 1

        """
        # initiate index vectors
        index_seq = []
        x1_train, x1_test, y1_train, y1_test = train_test_split(fw_fasta, readout, test_size=0.1, random_state=545)
        x2_train, x2_test, y2_train, y2_test = train_test_split(rc_fasta, readout, test_size=0.1, random_state=545)
        """

        # save the information of 10 folds auc scores
        train_auc_scores = []
        test_auc_scores = []

        for i in range(len(step)):
            if i == 0:
                x1_train = forward_shuffle[step[i]:]
                x1_test = forward_shuffle[0:step[i]]
                x2_train = reverse_shuffle[step[i]:]
                x2_test = reverse_shuffle[0:step[i]]
            else:
                x1_test = forward_shuffle[step[i-1]:step[i]]
                x1_train = forward_shuffle[0:step[i-1]]+forward_shuffle[step[i]:]
                x2_test = reverse_shuffle[step[i-1]:step[i]]
                x2_train = reverse_shuffle[0:step[i-1]]+reverse_shuffle[step[i]:]
            if i == 0:
                y1_train = readout_shuffle[step[i]:]
                y1_test = readout_shuffle[0:step[i]]
                #y2_train = readout_shuffle[step[i]:]
                #y2_test = readout_shuffle[0:step[i]]
            else:
                y1_test = readout_shuffle[step[i-1]:step[i]]
                y1_train = readout_shuffle[0:step[i-1]]+readout_shuffle[step[i]:]
                #y2_test = readout_shuffle[step[i-1]:step[i]]
                #y2_train = readout_shuffle[0:step[i-1]]+readout_shuffle[step[i]:]

            if i == 10:
                print("i was 10")
                x1_train = forward_shuffle[0:step[i-1]]
                x1_test = forward_shuffle[step[i-1]:step[i]]
                x2_train = reverse_shuffle[0:step[i-1]]
                x2_test = reverse_shuffle[step[i-1]:step[i]]
                y1_train = readout_shuffle[0:step[i-1]]
                y1_test = readout_shuffle[step[i-1]:step[i]]

            # change to ndarray type to pass, y1_test = y2_test
            x1_train = np.array(x1_train)
            x1_test = np.array(x1_test)
            y1_train = np.array(y1_train)
            y1_test = np.array(y1_test)
            x2_train = np.array(x2_train)
            x2_test = np.array(x2_test)

            callback = EarlyStopping(monitor='val_coeff_determination', patience=5, mode='max')
            #with early stopping
            #history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=self.epochs, batch_size=self.batch_size, callbacks = [callback])
            #without early stopping
            model = self.create_model()

            model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)

            pred_train = model.predict({'forward': x1_test, 'reverse': x2_test})
            vals = []

            for i in range(len(pred_train)):
                if pred_train[i] < 0.5:
                    val = 0
                    vals.append(val)
                if pred_train[i] >= 0.5:
                    val = 1
                    vals.append(val)

            print(y1_train[0:10])
            print(vals[0:10])

            true_pred = 0
            false_pred = 0
            for ind in range(len(pred_train)):
                if y1_train[ind] == vals[ind]:
                    true_pred += 1
                else:
                    false_pred += 1
            print('Total number of train-set predictions is: ' + str(len(y1_train)))
            print('Number of correct train-set predictions is: ' + str(true_pred))
            print('Number of incorrect train-set predictions is: ' + str(false_pred))

            auc_score = sklearn.metrics.roc_auc_score(y1_train, pred_train)
            print('train-set auc score is: ' + str(auc_score))
            print('train-set seed number is: ' + str(seed))
            train_auc_scores.append(auc_score)

            ##########################################################

            pred = model.predict({'forward': x1_test, 'reverse': x2_test})

            vals = []
            for i in range(len(pred)):
                if pred[i] < 0.5:
                    val = 0
                    vals.append(val)
                if pred[i] >= 0.5:
                    val = 1
                    vals.append(val)


            true_pred = 0
            false_pred = 0
            for ind in range(len(y1_test)):
                if y1_test[ind] == vals[ind]:
                    true_pred += 1
                else:
                    false_pred += 1
            print('Total number of test-set predictions is: ' + str(len(y1_test)))
            print('Number of correct test-set predictions is: ' + str(true_pred))
            print('Number of incorrect test-set predictions is: ' + str(false_pred))

            auc_score = sklearn.metrics.roc_auc_score(y1_test, pred)
            print('test-set auc score is: ' + str(auc_score))
            print('test-set seed number is: ' + str(seed))
            test_auc_scores.append(auc_score)

        print('seed number = %d' %seed)
        print(train_auc_scores)
        print('Mean train auc_scores of 10-fold cv is ' + str(np.mean(train_auc_scores)))
        print(test_auc_scores)
        print('Mean test auc_scores of 10-fold cv is ' + str(np.mean(test_auc_scores)))
