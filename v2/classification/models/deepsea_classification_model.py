
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
import pandas
import tensorflow as tf
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
# create model

def create_model(dim_num,parameters):

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












































































































































######################################################################################################
######################################################################################################
######################################################################################################
# RUN SCRIPT

run_model()

#nohup python deepsea.py sequences.fa wt_readout.dat parameters_deepsea.txt > outs/deepsea.out &