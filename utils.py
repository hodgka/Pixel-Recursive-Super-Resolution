import glob
import logging
import os
from datetime import datetime

import numpy as np
import scipy.misc
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
<<<<<<< HEAD
=======

>>>>>>> 1c32f535593de7483faf377efd979be849233740


def prepare_directories(delete_training_dir=False):
    """
    makes sure flagged directories are ready for I/O and gets list a dataset filenames
    Args:
        delete_training_dir - boolean, if truthy, deletes training_directory and remakes and empty dir
    returns:
        shuffled list of filenames, ready to be parsed
    """
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    if delete_training_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    if not tf.gfile.Exists(FLAGS.dataset) or not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder {}".format(FLAGS.dataset))

    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return filenames


def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):
    '''
    parse filenames into batches of image tensors and labels
    Args:
        sess - tensorflow session
        filenames - shuffled list of filenames to be parsed
        image_size - size of dataset images in pixels
        capacity_factor - number of batches allowed in file queue
    Returns:
        batches of parsed images and labels as tensors
    '''
    if not image_size:
        image_size = FLAGS.image_size
    reader = tf.WholeFileReader()
<<<<<<< HEAD
    filename_queue = tf.train.string_input_producer(filenames)

    k, v = reader.read(filename_queue)
    channels = 3
    image = tf.image.decode_jpeg(v, channels=channels, name='dataset_iamge')
    image.set_shape([None, None, channels])
=======
    height = width = 8
    os.chdir(data_path)
    # for root, dirs, files in os.walk(os.getcwd()):
    #     for f in files:
    #         if f.endswith('.jpg') or f.endswith('.jpeg'):
    #
    # image = transform.resize(io.imread(data_path))
    #


def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):
    if not image_size:
        image_size = FLAGS.image_size
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)

    k, v = reader.read(filename_queue)
    channels = 3
    image = tf.image.decode_jpeg(v, channels=channels, name='dataset_iamge')
    image.set_shape([None, None, channels])

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.95, 1.05)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.95, 1.05)

    wiggle = 8
    off_x, off_y = 25 - wiggle, 60 - wiggle
    crop_size = 128
    crop_size_plus = crop_size + 2 * wiggle
    image = tf.image.crop_to_bounding_box(image, off_x, off_y, crop_size_plus, crop_size_plus)
    image = tf.random_crop(image, [crop_size, crop_size, 3])

    image = tf.reshape(image, [1, crop_size, crop_size, 3])
    image = tf.cast(image, tf.float32) / 255.0

    if crop_size != image_size:
        image = tf.image.resize_area(image, [image_size, image_size])

    K = 4
    downsampled = tf.image.resize_area(image, [image_size // k, image_size // k])
    feature = tf.reshape(downsampled, [image_size // k, image_size // k])
    label = tf.reshape(image, [image_size, image_size, 3])

    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity=capacity_factor * FLAGS.batch_size,
                                      name='labels_and_features')
    tf.train.start_queue_runners(sess=sess)

    return features, labels
>>>>>>> 1c32f535593de7483faf377efd979be849233740

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.95, 1.05)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.95, 1.05)

    wiggle = 8
    off_x, off_y = 25 - wiggle, 60 - wiggle
    crop_size = 128
    crop_size_plus = crop_size + 2 * wiggle
    image = tf.image.crop_to_bounding_box(image, off_x, off_y, crop_size_plus, crop_size_plus)
    image = tf.random_crop(image, [crop_size, crop_size, 3])

    image = tf.reshape(image, [1, crop_size, crop_size, 3])
    image = tf.cast(image, tf.float32) / 255.0

    if crop_size != image_size:
        image = tf.image.resize_area(image, [image_size, image_size])

    K = 4
    downsampled = tf.image.resize_area(image, [image_size // k, image_size // k])
    feature = tf.reshape(downsampled, [image_size // k, image_size // k])
    label = tf.reshape(image, [image_size, image_size, 3])

    feature_batch, labels = tf.train.batch([feature, label],
                                           batch_size=FLAGS.batch_size,
                                           num_threads=4,
                                           capacity=capacity_factor * FLAGS.batch_size,
                                           name='labels_and_features')
    tf.train.start_queue_runners(sess=sess)

    return feature_batch, labels


def one_hot(y_i, size=256):
    """
    create a one-hot encoded vector , defaults to 256 entries
    args:
        y_i - class to be encoded
        size - size of vector. defaults to 256, for color inputs
    returns:
        one-hot vector
    """
    y_ = np.zeros(size, 1)
    y_[y_i] = 1
    return y_


def _summarize_progress(train_data, feature, label, gene_output, batch, suffix, max_samples=8):
    """
    create a summary of training progress
    Args:
        train_data - tensors from dataset
        feature
    """
    td = train_data

    size = [label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    bicubic = tf.image.resize_bicubic(feature, size)
    bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    image = tf.concat(2, [nearest, bicubic, clipped, label])

    image = image[0:max_samples, :, :, :]
    image = tf.concat(0, [image[i, :, :, :] for i in range(max_samples)])
    image = td.sess.run(image)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))


def _save_checkpoint(train_data, batch):
    """
    save network to be reused later
    """
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(FLAGS.checkpoint_dir, oldname)
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver()
    saver.save(td.sess, newname)

    print("    Checkpoint saved")
