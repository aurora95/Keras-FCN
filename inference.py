import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K

from models import *

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_name = 'AtrousFCN_Vgg16_16s'
    image_size = (320, 320)
    batch_shape = (1,)+ image_size + (3,)
    save_path = os.path.join(current_dir, 'Models/'+model_name)
    model_path = os.path.join(save_path, "model.json")
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')
    #model_path = os.path.join(current_dir, 'model_weights/fcn_atrous/model_change.hdf5')
    #model = FCN_Resnet50_32s((480,480,3))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    model = globals()[model_name](0.0, batch_shape=batch_shape)
    model.load_weights(checkpoint_path, by_name=True)

    '''from train import *
    model = load_model(model_path,
                        custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D, 'RecurrentContextModule': RecurrentContextModule, 'ContextCRF': ContextCRF,
                        'softmax_sparse_crossentropy_ignoring_last_label':softmax_sparse_crossentropy_ignoring_last_label,
                        'sparse_accuracy_ignoring_last_label': sparse_accuracy_ignoring_last_label})'''
    model.summary()

    img_nums = sys.argv[1:]#'2007_000491'
    for img_num in img_nums:
        image = cv2.imread('/home/aurora/Learning/Data/VOC2012/JPEGImages/%s.jpg'%img_num)
        label = Image.open('/home/aurora/Learning/Data/VOC2012/SegmentationClass/%s.png'%img_num)
        label_size = label.size

        img_h, img_w = image.shape[0:2]

        long_side = max(img_h, img_w, image_size[0], image_size[1])
        pad_w = long_side - img_w
        pad_h = long_side - img_h
        #image = np.lib.pad(image, ((pad_h/2, pad_h - pad_h/2), (pad_w/2, pad_w - pad_w/2), (0, 0)), 'constant')
        image = cv2.resize(image, image_size)

        image = np.expand_dims(image, axis=0)

        result = model.predict(image,batch_size=1)
        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)

        temp = Image.fromarray(result, mode='P')
        temp.palette = label.palette
        temp = temp.resize(label_size, resample=Image.BILINEAR)
        #temp = temp.crop((pad_w/2, pad_h/2, pad_w/2+img_w, pad_h/2+img_h))
        temp.show(title='result')
        print result
        print np.max(result)
