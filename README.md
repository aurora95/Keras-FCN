

Keras-FCN
---------

Fully convolutional networks and semantic segmentation with Keras.

![Biker Image](doc/2007_000129.jpg)

![Biker Ground Truth](doc/2007_000129.png)

![Biker as classified by AtrousFCN_Resnet50_16s](doc/AtrousFCN_Resnet50_16s_2007_000129.png)

## Models

Models are found in [models.py](models.py), and include ResNet and DenseNet based models. `AtrousFCN_Resnet50_16s` is the current best performer, with pixel mean Intersection over Union `mIoU 0.661076`, and pixel accuracy around `0.9` on the augmented Pascal VOC2012 dataset detailed below.

## Install

Useful setup scripts for Ubuntu 14.04 and 16.04 can be found in the [robotics_setup](https://github.com/ahundt/robotics_setup) repository. First use that to install CUDA, TensorFlow,

```
mkdir -p ~/src

cd ~/src
# install dependencies
pip install pillow keras sacred

# fork of keras-contrib necessary for DenseNet based models
git clone git@github.com:ahundt/keras-contrib.git -b densenet-atrous
cd keras-contrib
sudo python setup.py install


# Install python coco tools
cd ~/src
git clone https://github.com/pdollar/coco.git
cd coco
sudo python setup.py install

cd ~/src
git clone https://github.com/aurora95/Keras-FCN.git
```

## Datasets

Datasets can be downloaded and configured in an automated fashion via the ahundt-keras branch on a fork of the [tf_image_segmentation](https://github.com/ahundt/tf-image-segmentation/tree/ahundt-keras) repository.

For simplicity, the instructions below assume all repositories are in `~/src/`, and datasets are downloaded to `~/.keras/` by default.

```
cd ~/src
git clone git@github.com:ahundt/tf-image-segmentation.git -b Keras-FCN
```

### Pascal VOC + Berkeley Data Augmentation

[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) augmented with [Berkeley Semantic Contours](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/) is the primary dataset used for training Keras-FCN. Note that the default configuration maximizes the size of the dataset, and will not in a form that can be submitted to the pascal [VOC2012 segmentation results leader board](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6), details are below.


```
# Automated Pascal VOC Setup (recommended)
export PYTHONPATH=$PYTHONPATH:~/src/tf-image-segmentation
cd path/to/tf-image-segmentation/tf_image_segmentation/recipes/pascal_voc/
python data_pascal_voc.py pascal_voc_setup
```

This downloads and configures image/annotation filenames pairs train/val splits from combined Pascal VOC with train and validation split respectively that has
image full filename/ annotation full filename pairs in each of the that were derived
from PASCAL and PASCAL Berkeley Augmented dataset.

The datasets can be downloaded manually as follows:

```
# Manual Pascal VOC Download (not required)

    # original PASCAL VOC 2012
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB
    # berkeley augmented Pascal VOC
    wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB
```

The setup utility has three type of train/val splits(credit matconvnet-fcn):

    Let BT, BV, PT, PV, and PX be the Berkeley training and validation
    sets and PASCAL segmentation challenge training, validation, and
    test sets. Let T, V, X the final trainig, validation, and test
    sets.
    Mode 1::
          V = PV (same validation set as PASCAL)
    Mode 2:: (default))
          V = PV \ BT (PASCAL val set that is not a Berkeley training
          image)
    Mode 3::
          V = PV \ (BV + BT)
    In all cases:
          S = PT + PV + BT + BV
          X = PX  (the test set is uncahgend)
          T = (S \ V) \ X (the rest is training material)


### MS COCO


[MS COCO](mscoco.org) support is very experimental, contributions would be highly appreciated.

Note that there any pixel can have multiple classes, for example a pixel which is point on a cup on a table will be classified as both cup and table, but sometimes the z-ordering is wrong in the dataset. This means saving the classes as an image will result in very poor performance.

```
export PYTHONPATH=$PYTHONPATH:~/src/tf-image-segmentation
cd ~/src/tf-image-segmentation/tf_image_segmentation/recipes/mscoco

# Initial download is 13 GB
# Extracted 91 class segmentation encoding
# npy matrix files may require up to 1TB

python data_coco.py coco_setup
python data_coco.py coco_to_pascal_voc_imageset_txt
python data_coco.py coco_image_segmentation_stats

# Train on coco
cd ~/src/Keras-FCN
python train_coco.py
```


## Training and testing

The default configuration trains and evaluates `AtrousFCN_Resnet50_16s` on pascal voc 2012 with berkeley data augmentation.

```
cd ~/src/Keras-FCN
cd utils

# Generate pretrained weights
python transfer_FCN.py

cd ~/src/Keras-FCN

# Run training
python train.py

# Evaluate the performance of the network
python evaluate.py

```

Model weights will be in `~/src/Keras-FCN/Models`, along with saved image segmentation results from the validation dataset.