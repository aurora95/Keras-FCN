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
    #label.show(title='ground truth')
    #label = img_to_array(label)
    image = cv2.resize(image, (224, 224),interpolation=cv2.INTER_AREA)
    #cv2.imshow('label', label)
    #cv2.waitKey(60)
    image = np.expand_dims(image, axis=0)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'FCN_Resnet50_32s')
    model_path = os.path.join(save_path, "model.json")
    f = open(model_path, 'r')
    model_json = f.read()
    f.close
    model = model_from_json(model_json)
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')
    model.load_weights(checkpoint_path)
    model.summary()

    result = model.predict(image,batch_size=1)
    result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)
    
    temp = label.copy()
    temp.putdata(result)
    #temp.show(title='result')
    print result
    print np.max(result)
