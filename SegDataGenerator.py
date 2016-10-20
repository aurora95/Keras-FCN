from keras.preprocessing.image import *
from PIL import Image
import cv2
import numpy as np
import os

class SegDirectoryIterator(Iterator):
    '''
    Users need to ensure that all files exist.
    Label images should be png images where pixel values represents class number.
    '''
    def __init__(self, file_path, seg_data_generator,
                 data_dir, data_suffix,
                 label_dir, label_suffix, nb_classes, ignore_label=255,
                 target_size=None, color_mode='rgb',
                 dim_ordering='default', class_mode='sparse',
                 batch_size=1, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.file_path = file_path
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.nb_classes = nb_classes
        self.seg_data_generator = seg_data_generator
        self.target_size = tuple(target_size)
        self.ignore_label = ignore_label
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        self.nb_label_ch = 1
        if target_size:
            if self.color_mode == 'rgb':
                if self.dim_ordering == 'tf':
                    self.image_shape = self.target_size + (3,)
                else:
                    self.image_shape = (3,) + self.target_size
            else:
                if self.dim_ordering == 'tf':
                    self.image_shape = self.target_size + (1,)
                else:
                    self.image_shape = (1,) + self.target_size
            if self.dim_ordering == 'tf':
                self.label_shape = self.target_size + (self.nb_label_ch,)
            else:
                self.label_shape = (self.nb_label_ch,) + self.target_size
        elif batch_size != 1:
            raise ValueError('Batch size must be 1 when target image size is undetermined')
        else:
            self.image_shape = None
            self.label_shape = None
        if class_mode not in {'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of '
                             '"sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        #build lists for data files and label files
        self.data_files = []
        self.label_files= []
        fp = open(file_path)
        lines = fp.readlines()
        fp.close( )
        self.nb_sample = len(lines)
        for line in lines:
            line = line.strip('\n')
            self.data_files.append(line + data_suffix)
            self.label_files.append(line + label_suffix)
        super(SegDirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        if self.target_size:
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            batch_y = np.zeros((current_batch_size,) + self.label_shape)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data and labels
        for i, j in enumerate(index_array):
            data_file = self.data_files[j]
            label_file= self.label_files[j]
            img = load_img(os.path.join(self.data_dir, data_file), grayscale=grayscale, target_size=self.target_size)
            label = Image.open(os.path.join(self.label_dir, label_file))
            if self.target_size:
                label = label.resize((self.target_size[1], self.target_size[0]))
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            y = img_to_array(label, dim_ordering=self.dim_ordering).astype(int)
            if self.ignore_label:
                y[np.where(y==self.ignore_label)] = 0

            if self.target_size == None:
                batch_x = np.zeros((current_batch_size,) + x.shape)
                batch_y = np.zeros((current_batch_size,) + y.shape)
            x, y = self.seg_data_generator.random_transform(x, y)
            x = self.seg_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = y
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # return
        if self.class_mode == 'sparse':
            return batch_x, batch_y
        else:
            return batch_x

class SegDataGenerator(object):

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 channelwise_center = False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='constant',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 dim_ordering='default'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.ch_mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def flow_from_directory(self, file_path, data_dir, data_suffix,
                            label_dir, label_suffix, nb_classes, ignore_label=255,
                            target_size=(256, 256), color_mode='rgb',
                            class_mode='sparse',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg'):
        return SegDirectoryIterator(
            file_path, self,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix, nb_classes=nb_classes, ignore_label=ignore_label,
            target_size=target_size, color_mode=color_mode,
            dim_ordering=self.dim_ordering, class_mode=class_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def standardize(self, x):
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.channelwise_center:
            x -= self.ch_mean
        return x

    def random_transform(self, x, y):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        assert x.shape[img_row_index] == y.shape[img_row_index] and x.shape[img_col_index] == y.shape[
            img_col_index], 'DATA ERROR: Different shape of data and label!\ndata shape: %s, label shape: %s' % (str(x.shape), str(y.shape))

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode='constant', cval=0.0)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center and featurewise_std_normalization

        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

    def set_ch_mean(self, ch_mean):
        self.ch_mean = ch_mean
