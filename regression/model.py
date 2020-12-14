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

            alpha = 100
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
        #self.cross_val_custom()

    def create_model(self):
        # different metric functions
        def coeff_determination(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res/(SS_tot + K.epsilon()))
        def spearman_fn(y_true, y_pred):
            return tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout=tf.float32)

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
        first_layer = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', use_bias = True)

        fw = first_layer(forward)
        bw = first_layer(reverse)

        concat = concatenate([fw, bw], axis=1)

        concat_relu = ReLU()(concat)

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

        if self.regularizer == 'L_1':
            outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(flat)
        elif self.regularizer == 'L_2':
            outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation= self.activation_type)(flat)
        else:
            sys.exit()

        #weight_forwardin_0=model.layers[0].get_weights()[0]
        #print(weight_forwardin_0)
        model = keras.Model(inputs=[forward, reverse], outputs=outputs, name='mpra_model')
        model2 = keras.Model(inputs=[forward, reverse], outputs=pool_layer, name='intermediate_model')
        #print model summary
        model.summary()
        #sys.exit()

        model.compile(loss='mean_squared_error', optimizer='adam', metrics = [coeff_determination, spearman_fn])
        model2.compile(loss='mean_squared_error', optimizer='adam', metrics = [coeff_determination, spearman_fn])

        #return model
        return model, model2, first_layer

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

        if self.activation_type == 'linear':
            readout = np.log2(readout)
            readout = np.ndarray.tolist(readout)

        seed = random.randint(1,1000)

        x1_train, x1_test, y1_train, y1_test = train_test_split(fw_fasta, readout, test_size=0.1, random_state=seed)
        # split for reverse complemenet sequences
        x2_train, x2_test, y2_train, y2_test = train_test_split(rc_fasta, readout, test_size=0.1, random_state=seed)
        #assert x1_test == x2_test
        #assert y1_test == y2_test

        #model = self.create_model()
        model, model2, first_layer = self.create_model()

        # filters before training the model
        initial_weight = first_layer.get_weights()

        # change from list to numpy array
        y1_train = np.asarray(y1_train)
        y1_test = np.asarray(y1_test)
        y2_train = np.asarray(y2_train)
        y2_test = np.asarray(y2_test)

        # if we want to merge two training dataset
        # comb = np.concatenate((y1_train, y2_train))

        # train the data
        history = model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
        #history_ver2 = model2.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
        #with early stopping
        callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        #history = model.fit(features2, readout, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks = [callback])

        history2 = model.evaluate({'forward': x1_test, 'reverse': x2_test}, y1_test)
        pred = model.predict({'forward': x1_test, 'reverse': x2_test})

        print('metric values of model.evaluate: '+str(history2))
        print('metrics names are ' + str(model.metrics_names))

    def cross_val_custom(self):
        # so that we get different metrics used in this custom version
        # preprocess the data
        prep = preprocess(self.fasta_file, self.readout_file)

        # if want mono-nucleotide sequences
        dict = prep.one_hot_encode()
        # if want dinucleotide sequences
        #dict = prep.dinucleotide_encode()

        np.set_printoptions(threshold=sys.maxsize)

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        if self.activation_type == 'linear':
            readout = np.log2(readout)
            readout = np.ndarray.tolist(readout)

        step = list(range(0,len(fw_fasta), int(len(fw_fasta)/10)))
        step.append(len(fw_fasta))
        step.remove(0)

        # seed to reproduce results
        #seed = random.randint(1,1000)
        seed = 916
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
        true_vals = []
        pred_vals = []

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
            model, model2, first_layer = self.create_model()

            history = model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)
            history2 = model.evaluate({'forward': x1_test, 'reverse': x2_test}, y1_test)
            pred = model.predict({'forward': x1_test, 'reverse': x2_test})
            y_true2 = np.asarray(y1_test)

            # reshape as a column vector to concatenate
            y_true3 = np.reshape(y_true2, (len(y_true2), -1))

            true_pred = np.concatenate((y_true3,pred), axis=1)

            #np.set_printoptions(threshold=sys.maxsize)
            #np.savetxt(os.path.join('./output_files', 'y_true_%d_%d'%(seed,i)+'.txt'), y_true2)
            #np.savetxt(os.path.join('./output_files', 'y_pred_%d_%d'%(seed,i)+'.txt'), pred)
            np.savetxt(os.path.join('./problematic_seqs/output_files', 'vals_%d_%d'%(seed,i)+'.txt'), true_pred)

            y_pred = np.ndarray.tolist(pred)
            y_true = y1_test

            metrics.append((history2))

            # print individual plots and save

            #with open('y_true_%d_%d'%(seed, i)+'.txt', 'w') as file:
            #    file.write('y_true is \n' + str(y_true))
            #with open('y_pred_%d_%d'%(seed, i)+'.txt', 'w') as file:
            #    file.write('y_pred is \n' + str(y_pred))
            #with open('index_%d_%d'%(seed,i)+'.txt', 'w') as file:
            #    file.write('index of seqeucnes is \n' + str(index_seq))
            """
            #plt.subplot(int('34%d'%index))
            plt.subplot(3,4,index)
            plt.scatter(y_true, y_pred, s=1)
            index = index +1
            plt.grid(True)
            plt.xlim([-1, 3])
            plt.ylim([-1, 3])
            #plt.xlabel('y_true')
            #plt.ylabel('y_pred')
            """

        #plt.savefig('plot_%d'%seed)

        g1 = []
        g2 = []
        g3 = []
        for i in metrics:
            loss, r_2, spearman_val = i
            g1.append(loss)
            g2.append(r_2)
            g3.append(spearman_val)

        #plt.show()
        print(g2)
        print(g3)
        print('seed number = %d' %seed)
        print('Mean loss of 10-fold cv is ' + str(np.mean(g1)))
        print('Mean R_2 score of 10-fold cv is ' + str(np.mean(g2)))
        print('Mean Spearman of 10-fold cv is ' + str(np.mean(g3)))
