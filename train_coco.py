import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
import time
from keras.optimizers import SGD, Adam
from keras.callbacks import *
from keras.objectives import *
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util

from models import *
from train import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
# from tf_image_segmentation.recipes.mscoco import data_coco


if __name__ == '__main__':
    # model_name = 'AtrousFCN_Resnet50_16s'
    #model_name = 'Atrous_DenseNet'
    model_name = 'DenseNet_FCN'
    batch_size = 2
    batchnorm_momentum = 0.95
    epochs = 450
    lr_base = 0.2 * (float(batch_size) / 4)
    lr_power = float(1)/float(30)
    resume_training=False
    weight_decay = 0.0001/2
    target_size = (320, 320)
    dataset = 'COCO'
    if dataset == 'VOC2012_BERKELEY':
        # pascal voc + berkeley semantic contours annotations
        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/combined_annotations')
    if dataset is 'VOC2012':
        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')
        classes = 21
        class_weight = None
    elif dataset is 'COCO':
        train_file_path = os.path.expanduser('~/.keras/datasets/coco/annotations/train2014.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/coco/annotations/test2014.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/coco/train2014')
        label_dir       = os.path.expanduser('~/.keras/datasets/coco/seg_mask/train2014')
        stats_file = os.path.expanduser('~/.keras/datasets/coco/seg_mask/train2014/image_segmentation_class_stats.json')
        classes = 91
        # class_weight = data_coco.class_weight(image_segmentation_stats_file=stats_file)
        class_weight = None

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    train(batch_size, epochs, lr_base, lr_power, weight_decay, classes, model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=target_size, batchnorm_momentum=batchnorm_momentum, resume_training=resume_training,
          class_weight=class_weight, dataset=dataset)
