from keras.models import Sequential
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, BatchNormalization, MaxPooling1D
import keras

def create_model(self):
        model = model = Sequential()

        #First Conv1D
        model.add(Conv1D(filters=300, kernel_size=19, input_shape=(151,4)))

        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(pool_size=3))

        model.add(Conv1D(filters=200, kernel_size=11))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(pool_size=4))

        model.add(Conv1D(filters=200, kernel_size=7))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(rate=0.30))
        model.add(Dense(164, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            loss= "mean_squared_error",
            optimizer='adam')
        return model
