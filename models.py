import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
from keras_contrib.applications import densenet
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf

from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *


def top(x, input_shape, classes, activation, weight_decay):

    x = Conv2D(classes, (1, 1), activation='linear',
               padding='same', kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)

    if K.image_data_format() == 'channels_first':
        channel, row, col = input_shape
    else:
        row, col, channel = input_shape

    # TODO(ahundt) this is modified for the sigmoid case! also use loss_shape
    if activation is 'sigmoid':
        x = Reshape((row * col * classes,))(x)

    return x


def FCN_Vgg16_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
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
    x = Convolution2D(classes, 1, 1, init='he_normal', activation='linear', border_mode='valid', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)

    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model


def AtrousFCN_Vgg16_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
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

    # Convolutional layers transfered from fully-connected layers
    x = AtrousConvolution2D(4096, 7, 7, activation='relu', border_mode='same', atrous_rate=(2, 2),
                      name='fc1', W_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(4096, 1, 1, activation='relu', border_mode='same', name='fc2', W_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Convolution2D(classes, 1, 1, init='he_normal', activation='linear', border_mode='valid', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)

    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model


def FCN_Resnet50_32s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

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
    x = Convolution2D(classes, 1, 1, init='he_normal', activation='linear', border_mode='valid', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model


def AtrousFCN_Resnet50_16s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same', name='conv1', W_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), init='normal', activation='linear', border_mode='same', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)
    x = Convolution2D(classes, 1, 1, init='he_normal', activation='linear', border_mode='same', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)
    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model


def Atrous_DenseNet(input_shape=None, weight_decay=1E-4,
                    batch_momentum=0.9, batch_shape=None, classes=21,
                    include_top=False, activation='sigmoid'):
    # TODO(ahundt) pass the parameters but use defaults for now
    if include_top is True:
        # TODO(ahundt) Softmax is pre-applied, so need different train, inference, evaluate.
        # TODO(ahundt) for multi-label try per class sigmoid top as follows:
        # x = Reshape((row * col * classes))(x)
        # x = Activation('sigmoid')(x)
        # x = Reshape((row, col, classes))(x)
        return densenet.DenseNet(depth=None, nb_dense_block=3, growth_rate=32,
                                 nb_filter=-1, nb_layers_per_block=[6, 12, 24, 16],
                                 bottleneck=True, reduction=0.5, dropout_rate=0.2,
                                 weight_decay=1E-4,
                                 include_top=True, top='segmentation',
                                 weights=None, input_tensor=None,
                                 input_shape=input_shape,
                                 classes=classes, transition_dilation_rate=2,
                                 transition_kernel_size=(1, 1),
                                 transition_pooling=None)

    # if batch_shape:
    #     img_input = Input(batch_shape=batch_shape)
    #     image_size = batch_shape[1:3]
    # else:
    #     img_input = Input(shape=input_shape)
    #     image_size = input_shape[0:2]

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=16,
                                      data_format=K.image_data_format(),
                                      include_top=False)
    img_input = Input(shape=input_shape)

    x = densenet.__create_dense_net(classes, img_input,
                                    depth=None, nb_dense_block=3, growth_rate=32,
                                    nb_filter=-1, nb_layers_per_block=[6, 12, 24, 16],
                                    bottleneck=True, reduction=0.5, dropout_rate=0.2,
                                    weight_decay=1E-4, top='segmentation',
                                    input_shape=input_shape,
                                    transition_dilation_rate=2,
                                    transition_kernel_size=(1, 1),
                                    transition_pooling=None,
                                    include_top=include_top)

    x = top(x, input_shape, classes, activation, weight_decay)

    model = Model(img_input, x, name='Atrous_DenseNet')
    # TODO(ahundt) add weight loading
    return model


def DenseNet_FCN(input_shape=None, weight_decay=1E-4,
                 batch_momentum=0.9, batch_shape=None, classes=21,
                 include_top=False, activation='sigmoid'):
    if include_top is True:
        # TODO(ahundt) Softmax is pre-applied, so need different train, inference, evaluate.
        # TODO(ahundt) for multi-label try per class sigmoid top as follows:
        # x = Reshape((row * col * classes))(x)
        # x = Activation('sigmoid')(x)
        # x = Reshape((row, col, classes))(x)
        return densenet.DenseNetFCN(input_shape=input_shape,
                                    weights=None, classes=classes,
                                    nb_layers_per_block=[4, 5, 7, 10, 12, 15],
                                    growth_rate=16,
                                    dropout_rate=0.2)

    # if batch_shape:
    #     img_input = Input(batch_shape=batch_shape)
    #     image_size = batch_shape[1:3]
    # else:
    #     img_input = Input(shape=input_shape)
    #     image_size = input_shape[0:2]

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=16,
                                      data_format=K.image_data_format(),
                                      include_top=False)
    img_input = Input(shape=input_shape)

    x = densenet.__create_fcn_dense_net(classes, img_input,
                                        input_shape=input_shape,
                                        nb_layers_per_block=[4, 5, 7, 10, 12, 15],
                                        growth_rate=16,
                                        dropout_rate=0.2,
                                        include_top=include_top)

    x = top(x, input_shape, classes, activation, weight_decay)
    # TODO(ahundt) add weight loading
    model = Model(img_input, x, name='DenseNet_FCN')
    return model
