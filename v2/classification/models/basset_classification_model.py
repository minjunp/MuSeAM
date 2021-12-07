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
from tensorflow.keras import backend as K
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
# Create model

def create_model(dim_num,parameters):
    
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
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=parameters['loss_func'],
                  optimizer='adam', 
                  metrics=['binary_accuracy'])
    
    model.summary()

    return model


























































