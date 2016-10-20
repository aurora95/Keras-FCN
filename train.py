import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
from keras.optimizers import SGD
from keras.callbacks import *
from keras.objectives import *
import keras.utils.visualize_util as vis_util

from models import *
from SegDataGenerator import *

def train(batch_size, nb_epoch, lr_dict, weight_decay, nb_classes, model_name,
        train_file_path, val_file_path, data_dir, label_dir, target_size=None):
    if target_size:
        input_shape = target_size + (3,)
    else:
        input_shape = (None, None, 3)
    ################learning rate scheduler####################
    lr = 0.001
    def lr_scheduler(epoch):
        global lr
        if lr_dict.has_key(epoch):
            lr = lr_dict[epoch]
            print 'lr: %f' % lr
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)
    ###########################################################
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, model_name+'/')
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    ###########################################################
    tfboard = TensorBoard(log_dir=os.path.join(current_dir, save_path)+'/logs')
    ###########################################################
    model = globals()[model_name](input_shape, weight_decay)
    sgd = SGD(lr=0.001, momentum=0.9)
    model.compile(loss = softmax_sparse_crossentropy_ignoring_last_label, optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model_path = os.path.join(current_dir, save_path + "/model.json")
    # save model structure
    f = open(model_path, 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.close
    img_path = os.path.join(current_dir, save_path + "/model.png")
    vis_util.plot(model, to_file=img_path, show_shapes=True)

    # set data generator and train
    train_datagen = SegDataGenerator(channelwise_center=True, width_shift_range=0., height_shift_range=0.,
                           zoom_range=0.5, fill_mode='constant', cval=0., horizontal_flip=True, vertical_flip=False)
    train_datagen.set_ch_mean(np.array([104.00699, 116.66877, 122.67892]))
    val_datagen = SegDataGenerator(channelwise_center=True, fill_mode='constant')
    val_datagen.set_ch_mean(np.array([104.00699, 116.66877, 122.67892]))
    def get_file_len(file_path):
        fp = open(file_path)
        lines = fp.readlines()
        fp.close( )
        return len(lines)
    try:
        history = model.fit_generator(
                                    generator=train_datagen.flow_from_directory(
                                        file_path=train_file_path, data_dir=data_dir, data_suffix='.jpg',
                                        label_dir=label_dir, label_suffix='.png', nb_classes=nb_classes,
                                        target_size=target_size, color_mode='rgb',
                                        batch_size=batch_size, shuffle=True
                                    ),
                                    samples_per_epoch=get_file_len(train_file_path),
                                    nb_epoch=nb_epoch,
                                    callbacks=[scheduler, tfboard],
                                    validation_data=val_datagen.flow_from_directory(
                                        file_path=val_file_path, data_dir=data_dir, data_suffix='.jpg',
                                        label_dir=label_dir, label_suffix='.png', nb_classes=nb_classes,
                                        target_size=target_size, color_mode='rgb',
                                        batch_size=batch_size, shuffle=False
                                    ),
                                    nb_val_samples=100
                                )
    except Exception, e:
        print e
    finally:
        model.save(save_path + '/modelstate.h5', overwrite=True)
    model.save_weights(save_path+'/weights.h5')

if __name__ == '__main__':
    model_name = 'FCN_Vgg16_32s'
    batch_size = 1
    nb_epoch = 200
    lr_dict = lr_dict = {0: 0.00001, 80: 0.000001, 120: 0.0000001}
    weight_decay = 0.0002
    nb_classes = 21
    target_size = (224, 224)
    train_file_path = '/home/aurora/Learning/Data/VOC2012/ImageSets/Segmentation/train.txt'
    val_file_path   = '/home/aurora/Learning/Data/VOC2012/ImageSets/Segmentation/val.txt'
    data_dir        = '/home/aurora/Learning/Data/VOC2012/JPEGImages'
    label_dir       = '/home/aurora/Learning/Data/VOC2012/SegmentationClass'
    train(batch_size, nb_epoch, lr_dict, weight_decay, nb_classes, model_name, train_file_path, val_file_path, data_dir, label_dir, target_size=target_size)
