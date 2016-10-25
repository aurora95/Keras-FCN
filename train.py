import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD
from keras.callbacks import *
from keras.objectives import *
from keras.models import load_model
import keras.utils.visualize_util as vis_util

from models import *
from utils.loss_function import *
from utils.TrainingStateCheckpoint import *
from utils.SegDataGenerator import *

def sparse_accuracy_ignoring_last_label(labels, x):
    x = K.reshape(x, (-1, K.int_shape(x)[-1]))
    x = x + K.epsilon()
    softmax = K.softmax(x)

    labels = K.one_hot(tf.to_int32(K.flatten(labels)), K.int_shape(x)[-1]+1)
    labels = tf.pack(tf.unpack(labels, axis=-1)[:-1], axis=-1)

    return K.mean(K.equal(K.argmax(labels, axis=-1),
                    K.argmax(x, axis=-1)))

def train(batch_size, nb_epoch, lr_dict, weight_decay, nb_classes, model_name, train_file_path, val_file_path,
            data_dir, label_dir, target_size=None, resume_training=False):
    if target_size:
        input_shape = target_size + (3,)
    else:
        input_shape = (None, None, 3)

    ###########################################################
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, model_name)
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    ################learning rate scheduler####################
    lr = 0.
    def lr_scheduler(epoch):
        global lr
        if lr_dict.has_key(epoch):
            lr = lr_dict[epoch]
            print 'lr: %f' % lr
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)
    ######################## tfboard ###########################
    tfboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'))

    ####################### make model ########################
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')
    resume_from_epoch = 0
    if os.path.isfile(checkpoint_path) and resume_training: # resume from last checkpoint
        model_path = os.path.join(save_path, "model.json")
        f = open(model_path, 'r')
        model_json = f.read()
        f.close
        model = model_from_json(model_json, {'BilinearUpSampling2D': BilinearUpSampling2D})
        sgd = SGD(lr=0.001, momentum=0.9)
        model.compile(loss = softmax_sparse_crossentropy_ignoring_last_label, optimizer=sgd, metrics=[sparse_accuracy_ignoring_last_label])
        model.load_weights(checkpoint_path)
        with open(os.path.join(save_path, 'train_state.pkl')) as f:
            resume_from_epoch, batch_size, nb_epoch, lr_dict, weight_decay, nb_classes = pickle.load(f)

        print 'resuming from epoch %d'%resume_from_epoch
        print lr_dict
    else:
        model = globals()[model_name](input_shape, weight_decay)
        sgd = SGD(lr=0.001, momentum=0.9)
        model.compile(loss = softmax_sparse_crossentropy_ignoring_last_label, optimizer=sgd, metrics=[sparse_accuracy_ignoring_last_label])
        model_path = os.path.join(save_path, "model.json")
        # save model structure
        f = open(model_path, 'w')
        model_json = model.to_json()
        f.write(model_json)
        f.close
        img_path = os.path.join(save_path, "model.png")
        vis_util.plot(model, to_file=img_path, show_shapes=True)

    #################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
    train_state = TrainingStateCheckpoint(os.path.join(save_path, 'train_state.pkl'), batch_size, nb_epoch, lr_dict, weight_decay, nb_classes, resume_from_epoch)

    # set data generator and train
    train_datagen = SegDataGenerator(channelwise_center=True, zoom_range=0.5, fill_mode='constant', horizontal_flip=True)
    train_datagen.set_ch_mean(np.array([104.00699, 116.66877, 122.67892]))
    val_datagen = SegDataGenerator(channelwise_center=True, fill_mode='constant')
    val_datagen.set_ch_mean(np.array([104.00699, 116.66877, 122.67892]))
    def get_file_len(file_path):
        fp = open(file_path)
        lines = fp.readlines()
        fp.close( )
        return len(lines)
    history = model.fit_generator(
                                generator=train_datagen.flow_from_directory(
                                    file_path=train_file_path, data_dir=data_dir, data_suffix='.jpg',
                                    label_dir=label_dir, label_suffix='.png', nb_classes=nb_classes,
                                    target_size=target_size, color_mode='rgb',
                                    batch_size=batch_size, shuffle=True
                                ),
                                samples_per_epoch=get_file_len(train_file_path),
                                nb_epoch=nb_epoch,
                                callbacks=[scheduler, tfboard, checkpoint, train_state],
                                validation_data=val_datagen.flow_from_directory(
                                    file_path=val_file_path, data_dir=data_dir, data_suffix='.jpg',
                                    label_dir=label_dir, label_suffix='.png', nb_classes=nb_classes,
                                    target_size=target_size, color_mode='rgb',
                                    batch_size=batch_size, shuffle=False
                                ),
                                nb_val_samples=get_file_len(val_file_path)
                            )
    model.save(save_path+'/model.hdf5')

if __name__ == '__main__':
    model_name = 'FCN_Resnet50_32s'
    batch_size = 16
    nb_epoch = 200
    lr_dict = lr_dict = {0: 0.00001, 80: 0.000001, 120: 0.0000001}
    weight_decay = 0.0002
    nb_classes = 21
    target_size = (224, 224)
    train_file_path = '/home/aurora/Learning/Data/VOC2012/ImageSets/Segmentation/train.txt'
    val_file_path   = '/home/aurora/Learning/Data/VOC2012/ImageSets/Segmentation/val.txt'
    data_dir        = '/home/aurora/Learning/Data/VOC2012/JPEGImages'
    label_dir       = '/home/aurora/Learning/Data/VOC2012/SegmentationClass'
    train(batch_size, nb_epoch, lr_dict, weight_decay, nb_classes, model_name, train_file_path, val_file_path,
            data_dir, label_dir, target_size=target_size, resume_training=True)
