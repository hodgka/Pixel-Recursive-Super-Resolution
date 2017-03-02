import glob
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from skimage.io import imsave

FLAGS = tf.app.flags.FLAGS


def split_and_gate(tensor, channels):
    t1 = tensor[:, :, :, :channels]
    t2 = tensor[:, :, :, channels:]
    t1 = tf.nn.tanh(t1)
    t2 = tf.nn.sigmoid(t2)
    return t1 * t2


def logits_to_pixel(logits, mu=1.0):
    """
    Helper function to convert output logits from network into pixels
    """
    rebalanced_logits = logits * mu
    probabilities = tf.nn.softmax(rebalanced_logits)
    pixel_vals = np.arange(0, 256, dtype=np.float32)
    pixels = np.sum(probabilities * pixel_vals)
    return tf.floor(pixels)


def normalize_color(image):
    """
    Helper to rescale pixel color intensity to [-1, 1]
    """
    return image / 255.0 - 0.5


def save_samples(fetched_images, image_path):
    """
    Save image sampled from network
    """
    fetched_images = fetched_images.astype(np.uint8)
    n, h, w, _ = fetched_images.shape
    num = int(n**0.5)
    merged_image = np.zeros((n * h, n * w, 3), dtype=np.uint8)
    for i in range(num):
        for j in range(num):
            merged_image[i * h:(i + 1) * h, j * w: (j + 1) * w, :] = fetched_images[i * num + j, :, :, :]
    imsave(image_path, merged_image)
