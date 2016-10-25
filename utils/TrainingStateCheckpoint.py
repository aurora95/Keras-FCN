from keras.callbacks import *
import pickle

class TrainingStateCheckpoint(Callback):
    def __init__(self, save_path, batch_size, nb_epoch, lr_dict, weight_decay, nb_classes, init_epoch=0):
        self.__dict__.update(locals())

    def on_epoch_end(self, epoch, logs={}):
        with open(self.save_path, mode='w') as f:
            nb_epoch_trained = epoch+1
            temp = {}
            k_list = self.lr_dict.keys()
            k_list.sort()
            for k in k_list:
                x = self.lr_dict[k]
                k_temp = k - nb_epoch_trained
                if k_temp < 0:
                    k_temp = 0
                temp[k_temp] = x
            pickle.dump((nb_epoch_trained, self.batch_size, self.nb_epoch - nb_epoch_trained, temp, self.weight_decay, self.nb_classes), f)
