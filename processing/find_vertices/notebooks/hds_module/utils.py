import numpy as np
import tensorflow as tf

import os
import ntpath
from six.moves import cPickle as pickle

### Functions for getting array of directory paths and array of file paths
def get_dir_paths(root):
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, n))]

def get_file_paths(root):
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isfile(os.path.join(root, n))]

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

## Function for saving an object to a pickle file
def save_to_pickle(pickle_file, object, force=True):
    if os.path.exists(pickle_file) and not force:
        print(f'{pickle_file} already present, skipping pickling')
    else:
        try:
            with open(pickle_file, 'wb') as file:
                pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'Unable to save object to {pickle_file}: {e}')
            raise

class ShowBestEpochResult(tf.keras.callbacks.Callback):
    def __init__(self, on_param='val_loss', show_params=['val_loss', 'loss'], mode='min'):
        self.on_param    = on_param
        self.show_params = show_params
        self.mode        = mode

    def on_train_begin(self, logs=None):
        # The number of epoch at which the best value is found.
        self.best_epoch = 0

        # The best value found for the on_param.
        self.best = np.Inf
        if self.mode == 'max':
            self.best = np.NINF
            
        # The values on best epoch for each param in show_param.
        self.best_values = {}

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.on_param)
        save_as_best = (self.mode=='min' and current < self.best) or \
                       (self.mode=='max' and current > self.best)

        if save_as_best: 
            self.best = current
            for param in self.show_params:
                self.best_values[param] = logs.get(param)
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        print(f"BestEpochResult. Epoch: {self.best_epoch+1},", end=" ")

        for param in self.best_values:
            print(f"{param}: {self.best_values[param]:.5f},", end=" ")
        print(" ")
