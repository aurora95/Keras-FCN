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
import keras.backend as K
import keras.utils.visualize_util as vis_util

from models import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
import time
from evaluate import evaluate


def train(batch_size, nb_epoch, lr_base, lr_power, weight_decay, nb_classes, model_name, train_file_path, val_file_path,
            data_dir, label_dir, target_size=None, batchnorm_momentum=0.9, resume_training=False):
    if target_size:
        input_shape = target_size + (3,)
    else:
        input_shape = (None, None, 3)
    batch_shape = (batch_size,) + input_shape

    ###########################################################
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + model_name)
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    ################learning rate scheduler####################
    def lr_scheduler(epoch):
        '''if lr_dict.has_key(epoch):
            lr = lr_dict[epoch]
            print 'lr: %f' % lr'''
        lr = lr_base * ((1 - float(epoch)/nb_epoch) ** lr_power)
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)
    ######################## tfboard ###########################
    tfboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'))

    ####################### make model ########################
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')
    resume_from_epoch = 0

    model = globals()[model_name](weight_decay, batch_shape=batch_shape, batch_momentum=batchnorm_momentum)
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
    model.summary()

    #################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.hdf5'), save_weights_only=True)#.{epoch:d}

    # set data generator and train
    train_datagen = SegDataGenerator(channelwise_center=True, zoom_range=0.5, rotation_range=30., crop_mode='random', crop_size=target_size,
                                    width_shift_range=0.1, height_shift_range=0.1, fill_mode='constant', horizontal_flip=True)
    train_datagen.set_ch_mean(np.array([104.00699, 116.66877, 122.67892]))
    val_datagen = SegDataGenerator(channelwise_center=True, fill_mode='constant')
    val_datagen.set_ch_mean(np.array([104.00699, 116.66877, 122.67892]))
    def get_file_len(file_path):
        fp = open(file_path)
        lines = fp.readlines()
        fp.close()
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
                                callbacks=[scheduler, tfboard, checkpoint],
				                nb_worker=4
                            )

    model.save_weights(save_path+'/model.hdf5')

if __name__ == '__main__':
    model_name = 'FCN_Resnet50_32s'
    batch_size = 8
    batchnorm_momentum = 0.9
    nb_epoch = 200
    lr_base = 0.005
    lr_power = 0.9
    weight_decay = 0.0001
    nb_classes = 21
    target_size = (320, 320)
    train_file_path = '/home/aurora/Learning/Data/VOC2012/ImageSets/Segmentation/train.txt'
    val_file_path   = '/home/aurora/Learning/Data/VOC2012/ImageSets/Segmentation/val.txt'
    data_dir        = '/home/aurora/Learning/Data/VOC2012/JPEGImages'
    label_dir       = '/home/aurora/Learning/Data/VOC2012/SegmentationClass'
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    train(batch_size, nb_epoch, lr_base, lr_power, weight_decay, nb_classes, model_name, train_file_path, val_file_path,
            data_dir, label_dir, target_size=target_size, batchnorm_momentum=batchnorm_momentum, resume_training=False)
    evaluate(model_name, (505, 505), 21, 1)
