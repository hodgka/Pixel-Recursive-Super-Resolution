import os

import tensorflow as tf


class Dataset(object):
    """
    Class to create a image reader queue to batch dataset
    """

    def __init__(self, data_path, iterations, batch_size):

        # TODO make this more efficient so that it checks if filenames have been processed before
        # and reuses work that has already been done
        self.records = filename_writer(data_path)
        filename_queue = tf.train.string_input_producer(self.records)
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)

        # try to open jpeg or png. otherwise raise exception
        try:
            image = tf.image.decode_jpeg(image_file, 3)
        except InvalidArgumentError:
            try:
                image = tf.image.decode_png(image_file, 3)
            except InvalidArgumentError:
                raise
        hr_image = tf.image.resize_images(image, [32, 32])  # downsample image
        lr_image = tf.image.resize_images(image, [8, 8])  # REALLY downsample image
        hr_image = tf.cast(hr_image, tf.float32)
        lr_image = tf.cast(lr_image, tf.float32)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 400 * batch_size

        # batches images of shape [batch_size, 32, 32, 3],[batch_size, 8, 8, 3]
        self.hr_images, self.lr_images = tf.train.shuffle_batch([hr_image, lr_image], batch_size=batch_size,
                                                                min_after_dequeue=min_after_dequeue, capacity=capacity)


def filename_writer(data_path, fname=None):
    prefix_files = lambda x: os.path.abspath(data_path + '/' + x)
    records = list(map(prefix_files, os.listdir(data_path)))
    # with open(fname, 'w') as f:
    #     f.write('\n'.join(records))
    return records
