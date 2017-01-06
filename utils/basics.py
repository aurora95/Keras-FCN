from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
import tensorflow as tf

def conv_relu(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_relu'):
            x = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, bias=bias,
                                 init="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = Activation("relu")(x)
        return x
    return f

def conv_bn(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_bn'):
            x = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, bias=bias,
                                 init="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
        return x
    return f

def conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_bn_relu'):
            x = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, bias=bias,
                                 init="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
        return x
    return f

def bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('bn_relu_conv'):
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
            x = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, bias=bias,
                                 init="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
        return x
    return f

def atrous_conv_bn(nb_filter, nb_row, nb_col, atrous_rate=(2, 2), subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('atrous_conv_bn'):
            x = AtrousConvolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, atrous_rate=atrous_rate, subsample=subsample, bias=bias,
                                 init="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
        return x
    return f

def atrous_conv_bn_relu(nb_filter, nb_row, nb_col, atrous_rate=(2, 2), subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('atrous_conv_bn_relu'):
            x = AtrousConvolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, atrous_rate=atrous_rate, subsample=subsample, bias=bias,
                                 init="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
        return x
    return f
