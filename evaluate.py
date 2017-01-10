import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import time
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.optimizers import SGD
from utils.SegDataGenerator import *
from utils.loss_function import *
from utils.metrics import *

from models import *

def get_file_len(file_path):
    fp = open(file_path)
    lines = fp.readlines()
    fp.close()
    return len(lines)

def evaluate(model_name, target_size, nb_classes, batch_size, val_file_path, data_dir, label_dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    val_samples = 1448
    batch_shape = (batch_size,)+ target_size + (3,)
    save_path = os.path.join(current_dir, 'Models/'+model_name)
    model_path = os.path.join(save_path, "model.json")
    checkpoint_path = os.path.join(save_path, 'model.hdf5')

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    model = globals()[model_name](0.0, batch_shape=batch_shape)
    model.load_weights(checkpoint_path, by_name=True)

    #model.summary()
    print 'model loaded'

    val_datagen = SegDataGenerator(channelwise_center=True, fill_mode='constant', label_cval=255, crop_mode='center', crop_size=target_size)
    val_datagen.set_ch_mean(np.array([104.00699, 116.66877, 122.67892]))
    sgd = SGD(lr=0.0001, momentum=0.9)
    model.compile(loss = softmax_sparse_crossentropy_ignoring_last_label, optimizer=sgd, metrics=[sparse_accuracy_ignoring_last_label, mean_iou_ignoring_last_label])
    print 'model compiled'

    start_time = time.time()
    print model.evaluate_generator(
        generator=val_datagen.flow_from_directory(
            file_path=val_file_path, data_dir=data_dir, data_suffix='.jpg',
            label_dir=label_dir, label_suffix='.png',nb_classes=nb_classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=False
        ),
        val_samples = val_samples
    )
    duration = time.time() - start_time
    print '{}s used.\n'.format(duration)

if __name__ == '__main__':
    model_name = 'FeatureContext_Resnet50_16s'
    target_size = (512, 512)
    nb_classes = 21
    batch_size = 1
    val_file_path   = '/home/aurora/Learning/Data/VOC2012/ImageSets/Segmentation/val.txt'
    data_dir        = '/home/aurora/Learning/Data/VOC2012/JPEGImages'
    label_dir       = '/home/aurora/Learning/Data/VOC2012/SegmentationClass'
    evaluate(model_name, target_size, nb_classes, batch_size, val_file_path, data_dir, label_dir)
