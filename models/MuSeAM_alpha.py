import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import Dense, concatenate, GlobalMaxPool1D, Conv1D, ReLU
from tensorflow.keras import backend as K, regularizers
import keras
from scipy.stats import spearmanr, pearsonr

class ConvolutionLayer(Conv1D):
    def __init__(self, filters, alpha, beta,
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
        self.alpha = alpha
        self.beta = beta
        self.run_value = 1

    def call(self, inputs):

      ## shape of self.kernel is (12, 4, 512)
      ##the type of self.kernel is <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>
        if self.run_value > 2:

            x_tf = self.kernel  ##x_tf after reshaping is a tensor and not a weight variable :(
            x_tf = tf.transpose(x_tf, [2, 0, 1])

            alpha = int(self.alpha)
            beta = float(self.beta)

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

def create_model(self):
    def coeff_determination(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res/(SS_tot + K.epsilon()))
    def spearman_fn(y_true, y_pred):
        return tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
               tf.cast(y_true, tf.float32)], Tout=tf.float32)

    fw_input = keras.Input(shape=(171,4), name = 'forward')
    rc_input = keras.Input(shape=(171,4), name = 'reverse')

    customConv = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', use_bias = True, alpha=self.alpha, beta=self.beta)
    fw = customConv(fw_input)
    rc = customConv(rc_input)
    concat = concatenate([fw, rc], axis=1)

    activation = ReLU()(concat)

    globalPooling = GlobalMaxPool1D()(activation)
    outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation='linear')(globalPooling)

    model = keras.Model(inputs=[fw_input, rc_input], outputs=outputs)
    model.summary()
    #keras.utils.plot_model(model, "MuSeAM_regression.png")

    model.compile(loss= 'mean_squared_error',
                  optimizer= 'adam',
                  metrics = [coeff_determination, spearman_fn])

    return model