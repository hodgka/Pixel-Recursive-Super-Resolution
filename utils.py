import glob
import logging
import os
from datetime import datetime

import numpy as np
import scipy.misc
import tensorflow as tf


def load_images(data_path='', is_directory=True):
    images = []
    reader = tf.WholeFileReader()
    height = width = 8
    os.chdir(data_path)
    for root, dirs, files in os.walk(os.getcwd()):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.jpeg'):

    image = transform.resize(io.imread(data_path))


def get_batch(data, counter, batch_size):
    if (batch_size + 1) * counter >= data.shape[0]:
        counter = 0
    batch = data[batch_size * counter: batch_size * (counter + 1)]
    counter += 1
    return (batch, counter)


def choose_optimizer(config):
    opt = config.opt
    if optimizer.lower() == 'rms':
        return


def one_hot(y_i):
    y_ = np.zeros(256, 1)
    y_[y_i] = 1
    return y_
