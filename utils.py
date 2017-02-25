import logging
import numpy as np
import os
import scipy.misc
from datetime import datetime
import tensorflow as tf


def load_images(data_path='', is_directory=True):
    images = []
    reader = tf.WholeFileReader()
    height = width = 8


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
