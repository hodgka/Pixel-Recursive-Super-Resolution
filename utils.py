from __future__ import absolute_import, division, print_function

import glob
import logging
import os

import numpy as np
import tensorflow as tf
from skimage.io import imsave
from tqdm import tqdm


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Not using tf.nn.softmax because it throws an axis out of bounds error"""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)  # only difference


def logits_to_pixel(logits):
    """
    Helper function to convert output logits from network into pixels
    """
    probs = softmax(logits)
    pixel_dict = np.arange(0, 256, dtype=np.float32)
    pixels = np.sum(probs * pixel_dict, axis=1)
    return np.floor(pixels)


def normalize(image):
    """
    Helper to rescale pixel color intensity to [-1, 1]
    """
    return image / 128.0 - 0.5


def save_samples(images, image_path):
    """
    Save image sampled from network
    """
    images = images.astype(np.uint8)
    n, h, w, _ = images.shape
    num = int(n ** 0.5)
    merged_image = np.zeros((n * h, n * w, 3), dtype=np.uint8)
    for i in range(num):
        for j in range(num):
            merged_image[i * h:(i + 1) * h, j * w: (j + 1) * w, :] = images[i * num + j, :, :, :]
    # tf.summary.image(image_path, merged_image)
    imsave(image_path, merged_image)


def sample_from_model(sess, conditioning_logits, prior_logits, dataset, config, step=None):
    conditioning_logits = conditioning_logits
    prior_logits = prior_logits

    hr_imgs = dataset.hr_images
    lr_imgs = dataset.lr_images
    hr_imgs_, lr_imgs_ = sess.run([hr_imgs, lr_imgs])

    generated_hr_imgs = np.zeros((config.batch_size, config.image_size, config.image_size, 3), dtype=np.float32)
    conditioning_logits_ = sess.run(conditioning_logits, feed_dict={lr_imgs: lr_imgs_})

    for i in tqdm(range(config.image_size)):
        for j in tqdm(range(config.image_size)):
            for c in range(3):
                prior_logits_ = sess.run(prior_logits, feed_dict={hr_imgs: generated_hr_imgs})
                logits = conditioning_logits_[:, i, j, c * 256:(c + 1) * 256] + prior_logits_[:, i, j, c * 256:(c + 1) * 256]
                
                # add pixel to generated image
                generated_hr_imgs[:, i, j, c] = logits_to_pixel(logits)
            # print("generating pixel", i, j)

    save_samples(lr_imgs_, config.model_dir + '/lr_' + str(step) + '.jpg')
    save_samples(hr_imgs_, config.model_dir + '/hr_' + str(step) + '.jpg')
    save_samples(generated_hr_imgs, config.model_dir + '/generate_' + str(step) + '.jpg')
