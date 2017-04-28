import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import time
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import keras.backend as K

from models import *
from inference import inference


def calculate_iou(model_name, nb_classes, res_dir, label_dir, image_list):
    conf_m = zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    # mean_acc = 0.
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print('#%d: %s' % (total, img_num))
        pred = img_to_array(Image.open('%s/%s.png' % (res_dir, img_num))).astype(int)
        label = img_to_array(Image.open('%s/%s.png' % (label_dir, img_num))).astype(int)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        # acc = 0.
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', img_num)

        #    if l==p:
        #        acc+=1
        #acc /= flat_pred.shape[0]
        #mean_acc += acc
    #mean_acc /= total
    #print 'mean acc: %f'%mean_acc
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU


def evaluate(model_name, weight_file, image_size, nb_classes, batch_size, val_file_path, data_dir, label_dir,
          label_suffix='.png',
          data_suffix='.jpg'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'Models/'+model_name+'/res/')
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    fp = open(val_file_path)
    image_list = fp.readlines()
    fp.close()

    start_time = time.time()
    inference(model_name, weight_file, image_size, image_list, data_dir, label_dir, return_results=False, save_dir=save_dir,
              label_suffix=label_suffix, data_suffix=data_suffix)
    duration = time.time() - start_time
    print('{}s used to make predictions.\n'.format(duration))

    start_time = time.time()
    conf_m, IOU, meanIOU = calculate_iou(model_name, nb_classes, save_dir, label_dir, image_list)
    print('IOU: ')
    print(IOU)
    print('meanIOU: %f' % meanIOU)
    print('pixel acc: %f' % (np.sum(np.diag(conf_m))/np.sum(conf_m)))
    duration = time.time() - start_time
    print('{}s used to calculate IOU.\n'.format(duration))

if __name__ == '__main__':
    # model_name = 'Atrous_DenseNet'
    model_name = 'AtrousFCN_Resnet50_16s'
    # model_name = 'DenseNet_FCN'
    weight_file = 'checkpoint_weights.hdf5'
    # weight_file = 'model.hdf5'
    image_size = (512, 512)
    nb_classes = 21
    batch_size = 1
    dataset = 'VOC2012_BERKELEY'
    if dataset == 'VOC2012_BERKELEY':
        # pascal voc + berkeley semantic contours annotations
        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/combined_annotations')
        label_suffix = '.png'
    if dataset == 'COCO':
        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')
        label_suffix = '.npy'
    evaluate(model_name, weight_file, image_size, nb_classes, batch_size, val_file_path, data_dir, label_dir,
             label_suffix=label_suffix, data_suffix=data_suffix)
