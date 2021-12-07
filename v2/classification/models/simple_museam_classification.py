######################################################################################################
######################################################################################################
######################################################################################################
# IMPORT LIBRARIES
import time
import sys
import numpy as np
import sys
import math
import os
import json
import csv
import pandas
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from tensorflow.keras import backend as K
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
sys.path.append('../preprocessing/')
from data_preprocess import preprocess
from sklearn.utils import shuffle
import random
from sklearn.preprocessing import MinMaxScaler
#Tensorflow objects
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, BatchNormalization, Activation, concatenate, ReLU
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
#from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow import keras
from numpy import newaxis
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
######################################################################################################
######################################################################################################
######################################################################################################
#Reproducibility
seed = 460
np.random.seed(seed)
tf.random.set_seed(seed)

######################################################################################################
######################################################################################################
######################################################################################################
# SET TRAIN
# Get dictionary from text file
def train(file_name):
    dict = {}
    with open(file_name) as f:
        for line in f:
           (key, val) = line.split()
           dict[key] = val

    # change string values to integer values
    dict["filters"] = int(dict["filters"])
    dict["kernel_size"] = int(dict["kernel_size"])
    dict["epochs"] = int(dict["epochs"])
    dict["batch_size"] = int(dict["batch_size"])
    dict["validation_split"] = float(dict["validation_split"])    
    return dict

def run_model(argv = None):
    if argv is None:
        argv = sys.argv
        fasta_file_positive = argv[1]
        fasta_file_negative = argv[2]
        parameter_file = argv[3]

    ## excute the code
    start_time = time.time()

    parameters = train(parameter_file)

    cros_eval(parameters,fasta_file_positive,fasta_file_negative)

    # reports time consumed during execution (secs)
    print("--- %s seconds ---" % (time.time() - start_time))

######################################################################################################
######################################################################################################
######################################################################################################
# SET UTILS METRICS
@tf.function()
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

@tf.function()    
def spearman_fn(y_true, y_pred):
    return tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
           tf.cast(y_true, tf.float32)], Tout=tf.float32)
######################################################################################################
######################################################################################################
######################################################################################################
# SET CUSTOM LOSSES
@tf.function()
def rank_mse(yTrue, yPred):
  lambda_value=0.15
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

######################################################################################################
######################################################################################################
######################################################################################################
# SET MODEL CONSTRUCTION
class ConvolutionLayer(Conv1D):
    def __init__(self, 
                 filters,
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

class Museam:
    def __init__(self,
                 dim_num,
                 filters, 
                 kernel_size, 
                 pool_type, 
                 regularizer, 
                 activation_type, 
                 epochs,
                 batch_size, 
                 loss_func, 
                 optimizer,
                 model_name):

        """initialize basic parameters"""
        self.dim_num = dim_num
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_type = pool_type
        self.regularizer = regularizer
        self.activation_type = activation_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.model_name = model_name

    def create_model(self):

        dim_num = self.dim_num

        # Input Node
        forward = tf.keras.Input(shape=(dim_num[1],dim_num[2]), name = 'forward')
        reverse = tf.keras.Input(shape=(dim_num[1],dim_num[2]), name = 'reverse')

        # Multinomial Layer
        first_layer = ConvolutionLayer(filters=self.filters, 
                                       kernel_size=self.kernel_size, 
                                       strides=1, 
                                       data_format='channels_last', 
                                       use_bias = True)

        fw = first_layer(forward)
        bw = first_layer(reverse)

        # Concatenate both strands
        concat = concatenate([fw, bw], axis=1)
        pool_size_input = concat.shape[1]
        concat_relu = ReLU()(concat)

        #Pooling Layer
        if self.pool_type == 'Max':
            pool_layer = MaxPooling1D(pool_size=pool_size_input)(concat_relu)
            #pool_layer = MaxPooling1D(pool_size=12)(concat_relu)
        elif self.pool_type == 'Ave':
            pool_layer = AveragePooling1D(pool_size=pool_size_input)(concat_relu)
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
            raise NameError('Set the pooling layer name correctly')

        # Flatten Layer (None, 512)
        flat = Flatten()(pool_layer)
        if self.activation_type == 'linear':
            if self.regularizer == 'L_1':
                outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(flat)
            elif self.regularizer == 'L_2':
                outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation= self.activation_type)(flat)
            else:
                raise NameError('Set the regularizer name correctly')
        elif self.activation_type =='sigmoid':
            outputs = Dense(1, activation= self.activation_type)(flat)

        # Model Creation
        model = keras.Model(inputs=[forward, reverse], outputs=outputs)

        # Model Summary
        model.summary()
        if self.loss_func == 'mse':
            model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        elif self.loss_func == 'huber':
            loss_huber = keras.losses.Huber(delta=1)
            model.compile(loss=loss_huber, optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        elif self.loss_func == 'mae':
            loss_mae = keras.losses.MeanAbsoluteError()
            model.compile(loss=loss_mae, optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        elif self.loss_func == 'rank_mse':
            model.compile(loss=rank_mse, optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        elif self.loss_func == 'poisson':
            poisson_loss = keras.losses.Poisson()
            model.compile(loss=poisson_loss, optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        elif self.loss_func == 'binary_crossentropy':
            binary_crossentropy_loss = keras.losses.BinaryCrossentropy()
            model.compile(loss=binary_crossentropy_loss, optimizer=self.optimizer, metrics = ['binary_accuracy'])
        else:
            raise NameError('Unrecognized Loss Function')

        return model

######################################################################################################
######################################################################################################
######################################################################################################
# EVAL MODEL

def cros_eval(parameters,
              fasta_file_positive,
              fasta_file_negative):

    # Preprocess the data
    positive_control = preprocess(f'../data/{fasta_file_positive}','../data/wt_readout.dat')
    positive_control_names = positive_control.read_fasta_name_into_array()
    positive_control = positive_control.one_hot_encode()
  
    negative_control = preprocess(f'../data/{fasta_file_negative}','../data/wt_readout.dat')
    negative_control_names = negative_control.read_fasta_name_into_array()
    negative_control  = negative_control.one_hot_encode()
    
    features_forward = np.append(positive_control['forward'],negative_control['forward'],axis=0)
    features_reversed = np.append(positive_control['reverse'],negative_control['reverse'],axis=0)
    targets = np.append(np.ones(len(positive_control['forward'])), np.zeros(len(negative_control['forward'])))      
    names = np.append(positive_control_names, negative_control_names, axis=0)
    
    # Get dim
    dim_num = features_forward.shape
    
    # Shuffle the data
    features_forward_shuffle, features_reversed_shuffle, target_shuffle, names_shuffle = shuffle(features_forward,features_reversed, targets, names, random_state=seed)
    target_shuffle = np.array(target_shuffle)

    #initialize metrics to save values 
    metrics = []

    #Provides train/test indices to split data in train/test sets.
    kFold = StratifiedKFold(n_splits=10)
    ln = np.zeros(len(target_shuffle))
        
    pred_vals = pandas.DataFrame()
    cv_results =pandas.DataFrame()

    Fold=0
    model_name = parameters['model_name']

    for train, test in kFold.split(ln, ln):

        model = Museam(dim_num,
                      parameters['filters'], 
                      parameters['kernel_size'], 
                      parameters['pool_type'], 
                      parameters['regularizer'], 
                      parameters['activation_type'], 
                      parameters['epochs'],
                      parameters['batch_size'], 
                      parameters['loss_func'], 
                      parameters['optimizer'],
                      parameters['model_name']).create_model()
  

        # Get splits
        fwd_train = features_forward_shuffle[train]
        rc_train = features_reversed_shuffle[train]
        fwd_test = features_forward_shuffle[test]
        rc_test = features_reversed_shuffle[test]
        y_train = target_shuffle[train]
        y_test = target_shuffle[test]
        names_train = names_shuffle[test]
        names_test = names_shuffle[test]

        # Train model
        history = model.fit({'forward': fwd_train, 'reverse': rc_train}, 
                            y_train, 
                            epochs=parameters['epochs'], 
                            batch_size=parameters['batch_size'], 
                            validation_split=parameters['validation_split']
                            )

        # Get metrics
        loss, accuracy = model.evaluate({'forward': fwd_test, 'reverse': rc_test}, y_test)
        pred = model.predict({'forward': fwd_test, 'reverse': rc_test})
        pred = np.reshape(pred,len(pred))
        auc = sklearn.metrics.roc_auc_score(np.where(y_test>0.5, 1.0, 0.0), np.where(pred>0.5, 1.0, 0.0))

        # Temporary fold dataframes
        temp = pandas.DataFrame({'sequence_names':np.array(names_test).flatten(),
                                     'true_vals':np.array(y_test).flatten(),
                                     'pred_vals':np.array(pred).flatten()})                        
        temp['Fold'] = Fold

        temp2 = pandas.DataFrame({"Fold":[Fold],
                                  "Loss":[loss],
                                  "Accuracy":[accuracy],
                                  "AUC":[auc], 
        })

        Fold=Fold+1

        #append to main dataframe
        pred_vals = pred_vals.append(temp,ignore_index=True)
        cv_results = cv_results.append(temp2, ignore_index=True)

    
    pred_vals.to_csv(f'../outs/metrics/{model_name}.csv')


    #calculate mean accuracy across all folds
    mean_acc = cv_results['Accuracy'].mean()
    mean_auc = cv_results['AUC'].mean()
    cv_results = cv_results.append({'Fold':'All folds','Loss':'None','Accuracy':mean_acc,'AUC':mean_auc}, ignore_index=True)
    cv_results.to_csv(f'../outs/metrics/{model_name}_cv_results.csv')

######################################################################################################
######################################################################################################
######################################################################################################
# RUN SCRIPT

#nohup python museam_classification.py top_10percent.fa bottom_10percent.fa parameters/parameters_museam.txt > outs/logs/museam.out
























































































































































######################################################################################################
######################################################################################################
######################################################################################################
# RUN SCRIPT

run_model()

#nohup python deepsea.py sequences.fa wt_readout.dat parameters_deepsea.txt > outs/deepsea.out &