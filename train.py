import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD, Adam
from keras.callbacks import *
from keras.objectives import *
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util

from models import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
import time


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
        #lr = lr_base * ((1 - float(epoch)/nb_epoch) ** lr_power)
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch)
        print('lr: %f' % lr)
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)

    ####################### make model ########################
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')

    model = globals()[model_name](weight_decay=weight_decay, input_shape=input_shape, batch_momentum=batchnorm_momentum)
    # optimizer = SGD(lr=lr_base, momentum=0.9)
    optimizer = Adam(lr_base)
    model.compile(loss=softmax_sparse_crossentropy_ignoring_last_label,
                  optimizer=optimizer,
                  metrics=[sparse_accuracy_ignoring_last_label])
    if resume_training:
        model.load_weights(checkpoint_path, by_name=True)
    model_path = os.path.join(save_path, "model.json")
    # save model structure
    # f = open(model_path, 'w')
    # model_json = model.to_json()
    # f.write(model_json)
    # f.close
    # img_path = os.path.join(save_path, "model.png")
    # #vis_util.plot(model, to_file=img_path, show_shapes=True)
    # model.summary()

    # lr_reducer      = ReduceLROnPlateau(monitor=softmax_sparse_crossentropy_ignoring_last_label, factor=np.sqrt(0.1),
    #                                     cooldown=0, patience=15, min_lr=0.5e-6)
    # early_stopper   = EarlyStopping(monitor=sparse_accuracy_ignoring_last_label, min_delta=0.0001, patience=70)
    # callbacks = [early_stopper, lr_reducer]
    callbacks = [scheduler]

    ######################## tfboard ###########################
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'), histogram_freq=10, write_graph=True)
        callbacks.append(tensorboard)
    #################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.hdf5'), save_weights_only=True)#.{epoch:d}
    callbacks.append(checkpoint)
    # set data generator and train
    train_datagen = SegDataGenerator(zoom_range=[0.5, 2.0], zoom_maintain_shape=True,
                                    crop_mode='random', crop_size=target_size, #pad_size=(505, 505),
                                    rotation_range=0., shear_range=0, horizontal_flip=True,
                                    channel_shift_range=20.,
                                    fill_mode='constant', label_cval=255)
    val_datagen = SegDataGenerator()

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
                                    batch_size=batch_size, shuffle=True,
                                    #save_to_dir='Images/'
                                ),
                                samples_per_epoch=get_file_len(train_file_path),
                                nb_epoch=nb_epoch,
                                callbacks=callbacks,
				                # nb_worker=4,
                                # validation_data=val_datagen.flow_from_directory(
                                #     file_path=val_file_path, data_dir=data_dir, data_suffix='.jpg',
                                #     label_dir=label_dir, label_suffix='.png',nb_classes=nb_classes,
                                #     target_size=target_size, color_mode='rgb',
                                #     batch_size=batch_size, shuffle=False
                                # ),
                                # nb_val_samples = 64
                            )

    model.save_weights(save_path+'/model.hdf5')

if __name__ == '__main__':
    # model_name = 'AtrousFCN_Resnet50_16s'
    # model_name = 'Atrous_DenseNet'
    model_name = 'DenseNet_FCN'
    batch_size = 3
    batchnorm_momentum = 0.95
    nb_epoch = 350
    lr_base = 0.2 * (float(batch_size) / 4)
    lr_power = float(1)/float(30)
    resume_training=False
    weight_decay = 0.0001/2
    nb_classes = 21
    target_size = (320, 320)
    train_file_path = os.path.expanduser('~/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
    val_file_path   = os.path.expanduser('~/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
    data_dir        = os.path.expanduser('~/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
    label_dir       = os.path.expanduser('~/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    train(batch_size, nb_epoch, lr_base, lr_power, weight_decay, nb_classes, model_name, train_file_path, val_file_path,
            data_dir, label_dir, target_size=target_size, batchnorm_momentum=batchnorm_momentum, resume_training=resume_training)
