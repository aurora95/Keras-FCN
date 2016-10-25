import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
import keras.backend as K
import tensorflow as tf

from utils.get_weights_path import *

def resize_images_bilinear(X, height_factor, width_factor, dim_ordering):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'th' dim_ordering)
    - [batch, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if dim_ordering == 'th':
        original_shape = K.int_shape(X)
        new_shape = tf.shape(X)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif dim_ordering == 'tf':
        original_shape = K.int_shape(X)
        new_shape = tf.shape(X)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), dim_ordering='default', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.size = tuple(size)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            width = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            height = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.dim_ordering == 'tf':
            width = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            height = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        return resize_images_bilinear(x, self.size[0], self.size[1],
                               self.dim_ordering)

    def get_config(self):
        config = {'size': self.size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def FCN_Vgg16_32s(input_shape = None, weight_decay=0.):
    img_input = Input(shape=input_shape)
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', W_regularizer=l2(weight_decay))(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2', W_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1', W_regularizer=l2(weight_decay))(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2', W_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1', W_regularizer=l2(weight_decay))(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2', W_regularizer=l2(weight_decay))(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3', W_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1', W_regularizer=l2(weight_decay))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2', W_regularizer=l2(weight_decay))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3', W_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1', W_regularizer=l2(weight_decay))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2', W_regularizer=l2(weight_decay))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3', W_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Convolution2D(4096, 7, 7, activation='relu', border_mode='same', name='fc1', W_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(4096, 1, 1, activation='relu', border_mode='same', name='fc2', W_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Convolution2D(21, 1, 1, init='normal', activation='linear', border_mode='valid', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)

    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model

# The original help functions from keras does not have weight regularizers, so I modified them.
# Also, I changed these two functions into functional style
def identity_block(kernel_size, filters, stage, block, weight_decay=0.):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a', W_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                          border_mode='same', name=conv_name_base + '2b', W_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', W_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = merge([x, input_tensor], mode='sum')
        x = Activation('relu')(x)
        return x
    return f

def conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                          name=conv_name_base + '2a', W_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                          name=conv_name_base + '2b', W_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', W_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                                 name=conv_name_base + '1', W_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = merge([x, shortcut], mode='sum')
        x = Activation('relu')(x)
        return x
    return f

def FCN_Resnet50_32s(input_shape = None, weight_decay=0.):
    img_input = Input(shape=input_shape)
    bn_axis = 3

    x = Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same', name='conv1', W_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

    x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)
    #classifying layer
    x = Convolution2D(21, 1, 1, init='normal', activation='linear', border_mode='valid', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model
