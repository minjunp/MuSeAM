import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import Dense, concatenate, GlobalMaxPool1D, Conv1D
from tensorflow.keras import backend as K, regularizers
import keras

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
        print("self.run value is", self.run_value)
        if self.run_value > 2:

            x_tf = self.kernel  ##x_tf after reshaping is a tensor and not a weight variable :(
            x_tf = tf.transpose(x_tf, [2, 0, 1])

            alpha = 100
            beta = 1/alpha
            bkg = tf.constant([0.25, 0.25, 0.25, 0.25])
            bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(lambda x: tf.math.scalar_mul(beta, tf.subtract(tf.subtract(tf.subtract(tf.math.scalar_mul(alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis = 1), axis = 1)), tf.expand_dims(tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.subtract(tf.math.scalar_mul(alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis = 1), axis = 1))), axis = 1)), axis = 1)), tf.math.log(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]])))), x_tf)
            transf = tf.transpose(filt_list, [1, 2, 0])
            outputs = self._convolution_op(inputs, transf)
        else:
            outputs = self._convolution_op(inputs, self.kernel)


        self.run_value += 1
        return outputs

def create_model(self):
    fw_input = keras.Input(shape=(150,4), name = 'forward')
    rc_input = keras.Input(shape=(150,4), name = 'reverse')

    customConv = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', use_bias = True)
    fw = customConv(fw_input)
    rc = customConv(rc_input)
    concat = concatenate([fw, rc], axis=1)
    globalPooling = GlobalMaxPool1D()(concat)
    fc1 = Dense(32)(globalPooling)
    outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation='sigmoid')(fc1)

    model = keras.Model(inputs=[fw_input, rc_input], outputs=outputs)
    model.summary()
    model.compile(loss= 'binary_crossentropy',
                  optimizer= 'adam',
                  metrics = [tf.keras.metrics.AUC()])

    return model
