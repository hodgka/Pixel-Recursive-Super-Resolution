import os

import tensorflow as tf


class Dataset(object):
    """
    Class to create a image reader queue to batch dataset
    """

    def __init__(self, data_path, iterations, batch_size):

        # TODO make this more efficient so that it only processes filenames the first time you train the network
        self.records = []
        for fname in os.listdir(data_path):
            self.records.append(os.path.abspath(data_path + "/" + fname))
        filename_queue = tf.train.string_input_producer(self.records)
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file, 3)

        hr_image = tf.image.resize_images(image, [32, 32])
        lr_image = tf.image.resize_images(image, [8, 8])
        hr_image = tf.cast(hr_image, tf.float32)
        lr_image = tf.cast(lr_image, tf.float32)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 400 * batch_size
        self.hr_images, self.lr_images = tf.train.shuffle_batch([hr_image, lr_image], batch_size=batch_size,
                                                                min_after_dequeue=min_after_dequeue, capacity=capacity)
