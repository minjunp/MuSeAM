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


def create_model(dim_num,parameters):

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
    model.add(tf.keras.layers.Conv1D(filters=320,
                 kernel_size=8,
                 input_shape=(dim_num[1],dim_num[2])))

    model.add(tf.keras.layers.MaxPooling1D(pool_size=4,strides=4))
    model.add(tf.keras.layers.Dropout(rate=0.20))
    #Second Conv1D
    model.add(tf.keras.layers.Conv1D(filters=480,
                 kernel_size=8))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=4,strides=4))
    model.add(tf.keras.layers.Dropout(rate=0.20))
    #Third Conv1D
    model.add(tf.keras.layers.Conv1D(filters=960,
                 kernel_size=8))
    model.add(tf.keras.layers.Dropout(rate=0.50))
    #Dense Layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(925, activation='relu'))
    #Output Layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=parameters['loss_func'],
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    model.summary()

    return model


def cros_eval(parameters,fasta_file_positive,fasta_file_negative):
    # Preprocess the data
    positive_control = preprocess(f'dnase/top5k/{fasta_file_positive}','wt_readout.dat')
    positive_control_names = positive_control.read_fasta_name_into_array()
    positive_control = positive_control.one_hot_encode()


    negative_control = preprocess(f'dnase/top5k/{fasta_file_negative}','wt_readout.dat')
    negative_control_names = negative_control.read_fasta_name_into_array()
    negative_control  = negative_control.one_hot_encode()


    features = np.append(positive_control['forward'],negative_control['forward'],axis=0)
    targets = np.append(np.ones(len(positive_control['forward'])), np.zeros(len(negative_control['forward'])))
    names = np.append(positive_control_names, negative_control_names, axis=0)



    dim_num = features.shape


    features_shuffle, target_shuffle, names_shuffle = shuffle(features, targets, names, random_state=seed)

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
        model = None
        model = create_model(dim_num,parameters)

        fwd_train = features_shuffle[train]
        fwd_test = features_shuffle[test]


        y_train = target_shuffle[train]
        y_test = target_shuffle[test]

        names_train = names_shuffle[test]
        names_test = names_shuffle[test]

        history = model.fit(fwd_train, y_train, epochs=parameters['epochs'], batch_size=parameters['batch_size'], validation_split=parameters['validation_split'])
        loss, accuracy = model.evaluate(fwd_test, y_test)
        pred = model.predict(fwd_test)
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

run_model()
