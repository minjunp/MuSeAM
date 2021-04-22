from keras.models import Sequential
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras import backend as K, regularizers
import keras

def create_model(self):
        model = Sequential()

        model.add(Conv1D(120, 5, activation='relu', input_shape=(151, 4)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Conv1D(120, 5, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Conv1D(120, 5, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(12, activation='linear'))
        model.summary()

        model.compile(
            loss= "mean_squared_error",
            optimizer='adam')
        return model
