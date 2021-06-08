from keras.models import Sequential
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, BatchNormalization, MaxPooling1D
import keras

def create_model(self):
        model = Sequential()

        model.add(Conv1D(filters=320, kernel_size=8, input_shape=(151,4)))
        model.add(MaxPooling1D(pool_size=4,strides=4))
        model.add(Dropout(rate=0.20))

        model.add(Conv1D(filters=480, kernel_size=8))
        model.add(MaxPooling1D(pool_size=4,strides=4))
        model.add(Dropout(rate=0.20))

        model.add(Conv1D(filters=960, kernel_size=8))
        model.add(Dropout(rate=0.50))

        model.add(Flatten())
        model.add(Dense(925, activation='relu'))
        #Output Layer
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            loss= "mean_squared_error",
            optimizer='adam')
        return model
