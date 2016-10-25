from keras.objectives import *
import keras.backend as K
import tensorflow as tf

# Softmax cross-entropy loss function for segmentation
def softmax_sparse_crossentropy_ignoring_last_label(labels, x):
    x = K.reshape(x, (-1, K.int_shape(x)[-1]))
    x = x + K.epsilon()
    softmax = K.softmax(x)

    labels = K.one_hot(tf.to_int32(K.flatten(labels)), K.int_shape(x)[-1]+1)
    labels = tf.pack(tf.unpack(labels, axis=-1)[:-1], axis=-1)

    cross_entropy = -K.sum(labels * K.log(softmax), axis=1)
    cross_entropy_mean = K.mean(cross_entropy)
    return cross_entropy_mean
