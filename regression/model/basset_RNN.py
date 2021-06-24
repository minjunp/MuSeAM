
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras import backend as K
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from data_preprocess import preprocess
from sklearn.utils import shuffle
import random
from sklearn.preprocessing import MinMaxScaler
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
# SET FUNCTIONS

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
        #input argszw
        fasta_file = argv[1]
        #e.g. sequences.fa
        readout_file = argv[2]
        #e.g. wt_readout.dat
        parameter_file = argv[3]
        #e.g. parameter1.txt

    ## excute the code
    start_time = time.time()

    parameters = train(parameter_file)

    cros_eval(parameters,fasta_file,readout_file)

    # reports time consumed during execution (secs)
    print("--- %s seconds ---" % (time.time() - start_time))


def create_model(dim_num):

    # Modified custom metric functions
    def coeff_determination(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res/(SS_tot + K.epsilon()))
    def spearman_fn(y_true, y_pred):
        return tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),tf.cast(y_true, tf.float32)], Tout=tf.float32)
    
    #deepsea arquitecture
    model = tf.keras.Sequential()

    #First Conv1D
    model.add(tf.keras.layers.Conv1D(filters=300, 
                 kernel_size=19, 
                 input_shape=(dim_num[1],dim_num[2])))
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))

    ## model.add(tf.keras.layers.Dropout(rate=0.20))
    
    #Second Conv1D
    model.add(tf.keras.layers.Conv1D(filters=200, 
                 kernel_size=11))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=4))
    
    #Third Conv1D
    model.add(tf.keras.layers.Conv1D(filters=200, 
                 kernel_size=7))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    

    #Dense Layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.30))
    model.add(tf.keras.layers.Dense(164, activation='relu'))

    #Output Layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mse', 
                  optimizer='adam', 
                  metrics=[coeff_determination, spearman_fn])
    
    model.summary()

    return model


def cros_eval(parameters,fasta_file,readout_file):
    # Preprocess the data
    prep = preprocess(fasta_file, readout_file)
    dict = prep.one_hot_encode()
    
    fw_fasta = dict["forward"]
    rc_fasta = dict["reverse"]
    readout = dict["readout"]
    names = prep.read_fasta_name_into_array()


    dim_num = fw_fasta.shape


    readout = np.log2(readout)
    if parameters['scaling'] == 'no_scaling':
        readout = np.ndarray.tolist(readout)
    elif parameters['scaling'] == "0_1":
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(readout.reshape(-1, 1))
        readout = scaler.transform(readout.reshape(-1, 1))
        readout = readout.flatten()
        readout = np.ndarray.tolist(readout)
    elif parameters['scaling'] == "-1_1":
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(readout.reshape(-1, 1))
        readout = scaler.transform(readout.reshape(-1, 1))
        readout = readout.flatten()
        readout = np.ndarray.tolist(readout)



    forward_shuffle, readout_shuffle, names_shuffle = shuffle(fw_fasta, readout, names, random_state=seed)
    reverse_shuffle, readout_shuffle, names_shuffle = shuffle(rc_fasta, readout, names, random_state=seed)
    readout_shuffle = np.array(readout_shuffle)

    #initialize metrics to save values 
    metrics = []

    #Provides train/test indices to split data in train/test sets.
    kFold = StratifiedKFold(n_splits=10)
    ln = np.zeros(len(readout_shuffle))
        
    pred_vals = pandas.DataFrame()

    Fold=0
    model_name = parameters['model_name']

    for train, test in kFold.split(ln, ln):
        model = None
        model = create_model(dim_num)

        fwd_train = forward_shuffle[train]
        fwd_test = forward_shuffle[test]
        rc_train = reverse_shuffle[train]
        rc_test = reverse_shuffle[test]
        y_train = readout_shuffle[train]
        y_test = readout_shuffle[test]
        names_train = names_shuffle[test]
        names_test = names_shuffle[test]

        history = model.fit(fwd_train, y_train, epochs=parameters['epochs'], batch_size=parameters['batch_size'], validation_split=parameters['validation_split'])
        history2 = model.evaluate(fwd_test, y_test)
        pred = model.predict(fwd_test)

        metrics.append(history2)
        pred = np.reshape(pred,len(pred))

        temp = pandas.DataFrame({'sequence_names':np.array(names_test).flatten(),
                                     'true_vals':np.array(y_test).flatten(),
                                     'pred_vals':np.array(pred).flatten()})
        temp['Fold'] = Fold

        Fold=Fold+1

        pred_vals = pred_vals.append(temp,ignore_index=True)

    
    pred_vals.to_csv(f'./outs/{model_name}.csv')

    print('[INFO] Calculating 10Fold CV metrics')   
    g1 = []
    g2 = []
    g3 = []
    for i in metrics:
        loss, r_2, spearman_val = i
        g1.append(loss)
        g2.append(r_2)
        g3.append(spearman_val)

    print(g2)
    print(g3)
    print('seed number = %d' %seed)
    print('Mean loss of 10-fold cv is ' + str(np.mean(g1)))
    print('Mean R_2 score of 10-fold cv is ' + str(np.mean(g2)))
    print('Mean Spearman of 10-fold cv is ' + str(np.mean(g3)))

    metrics_dataframe = pandas.DataFrame({"mean_loss":[np.mean(g1)],
                                          "R_2":[np.mean(g2)],
                                          "Spearman":[np.mean(g3)]})


    metrics_dataframe.to_csv(f'./outs/{model_name}_CV_metrics.csv')



######################################################################################################
######################################################################################################
######################################################################################################
# RUN SCRIPT

run_model()

#nohup python basset.py sequences.fa wt_readout.dat parameters_basset.txt > outs/basset.out &