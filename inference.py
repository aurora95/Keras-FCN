import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model

from models import *

if __name__ == '__main__':
    image = cv2.imread('/home/aurora/Learning/Data/VOC2012/JPEGImages/2007_000033.jpg')
    label = Image.open('/home/aurora/Learning/Data/VOC2012/SegmentationClass/2007_000033.png')
    label = img_to_array(label)
    image = cv2.resize(image, (224, 224),interpolation=cv2.INTER_AREA)
    #cv2.imshow('label', label)
    #cv2.waitKey(60)
    image = np.expand_dims(image, axis=0)
    model = load_model('FCN_Vgg16_32s/modelstate.h5')
    #model = FCN_Vgg16_32s((224,224,3))
    model.summary()
    result = model.predict(image,batch_size=1)
    print result
    print result.shape
