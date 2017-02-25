from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf


class ConditionalNet:
    # ResNet
    pass


class GatedPixelCNN:
    """
    Gated PixelCNN implementation
    Methods:
        __init__:
    This class implements the Gated version of PixelCNN in the paper https: // arxiv.org/abs/1606.05328
    images : [batch_size, height, width, channels]
    output_images : [batch_size, height, width, channels, q_levels] if q_levels is not None
                    [batch_size, height, width, channels] otherwise
    """

    def __init__(self,
                 images,
                 input_dims,
                 activation="relu",
                 features=32,
                 q_levels=256,
                 filter_sizes=[7, 3],
                 num_layers=12,
                 strides=[1, 1]):
        '''
        Constructor that produces
        Args:
            Images:
            input_dims - Input dimensions of the images
            activation - type of activation as a string
            features -
            q_levels -
            filter_sizes -
            num_layers -
            strides -
        '''
        self.X = images
        self.activation = self.figure_out_act(activation)

        # Generating the initial vertical and horizontal stacks for the PixelCNN
        # First vertical and horizontal stacks have a filter size of filter_sizes[0]
        # and subsequent have a size of filter_sizes[1].
        # The padding has shape [row_add, columns_add] i.e row_add no of rows are added on top and bottom
        # of the image, columns_add no of coluumns are added on theleft and right of the image.

        v1 = self.get_conv_layer(self.X,
                                 input_dims,
                                 filter_size=(filter_sizes[0] // 2 + 1, filter_sizes[0]),
                                 padding=(filter_sizes[0] // 2 + 1, filter_sizes[0] // 2),
                                 out_channels=features,
                                 stride=strides,
                                 masking=None,
                                 activation=self.activation,
                                 scope="v1_stack")
        v1_shape = v1.get_shape().as_list()

        # While generating i th row we can only use vertical contribution till i-1
        # Note that with given padding v1(and all other v variables) has a dummy/mask row
        # in the beginning that depends on nothing.
        #
        # can only use vertical contribution up to i-1
        h_input = tf.concat(3, [tf.slice(v1, [0, 0, 0, 0], [v1_shape[0], v1_shape[
                            1] - filter_sizes[0] // 2 - 2, v1_shape[2], v1_shape[3]]), self.X])
        h_input_shape = h_input.get_shape().as_list()
        h1 = self.get_conv_layer(h_input,
                                 h_input_shape,
                                 filter_size=(1, filter_sizes[0]),
                                 padding=(0, filter_sizes[0] // 2),
                                 out_channels=features,
                                 stride=strides,
                                 masking=None,
                                 activation=self.activation,
                                 scope="h1_stack")

        # since 1st row of all v variables doesnt is like a mask and given the current padding,
        # there are some extra rows that turn up at the end so we need to slice the
        # x_v
        x_h = h1
        x_v = v1[:, 1:-(filter_sizes[0] // 2) - 1, :, :]

        for i in range(num_layers - 2):
            x_v_shape = x_v.get_shape().as_list()
            v_stack = self.get_conv_layer(x_v,
                                          x_v_shape,
                                          filter_size=(
                                              filter_sizes[1] // 2 + 1, filter_sizes[1]),
                                          padding=(filter_sizes[
                                                   1] // 2 + 1, filter_sizes[1] // 2),
                                          out_channels=features,
                                          stride=strides,
                                          masking=None,
                                          activation=self.activation,
                                          scope="v" + str(i + 2) + "_stack")

            v_stack_shape = v_stack.get_shape().as_list()
            v2v = self.get_conv_layer(v_stack,
                                      v_stack_shape,
                                      filter_size=(1, 1),
                                      padding=None,
                                      out_channels=features,
                                      stride=strides,
                                      masking=None,
                                      activation=self.activation,
                                      scope="v2v" + str(i + 2) + "_stack")
            v_shape = v2v.get_shape().as_list()

            h_input = tf.concat(3, [tf.slice(v2v, [0, 0, 0, 0], [v_shape[0], v_shape[1] -
                                                                 filter_sizes[1] // 2 - 2, v_shape[2], v_shape[3]]), x_h])
            h_input_shape = h_input.get_shape().as_list()
            h_stack = self.get_conv_layer(h_input,
                                          h_input_shape,
                                          filter_size=(1, filter_sizes[1] // 2 + 1),
                                          padding=(0, filter_sizes[1] // 2),
                                          out_channels=features,
                                          stride=strides,
                                          masking=None,
                                          activation=self.activation,
                                          scope="h" + str(i + 2) + "_stack")

            h2h_input = tf.tanh(v2v[:, :-(filter_sizes[1] // 2) - 2, :, :]) * \
                tf.sigmoid(h_stack[:, :, :-(filter_sizes[1] // 2), :])
            h2h_input_shape = h2h_input.get_shape().as_list()

            h2h = self.get_conv_layer(h2h_input,
                                      h2h_input_shape,
                                      filter_size=(1, 1),
                                      padding=None,
                                      out_channels=features,
                                      stride=strides,
                                      masking=None,
                                      activation=self.activation,
                                      scope="h2h" + str(i + 2) + "_stack")

            # residual connections are added to horizontal layer only
            x_h = h2h + x_h
            x_v = tf.tanh(v2v[:, 1:-(filter_sizes[1] // 2) - 1, :, :]) * \
                tf.sigmoid(v_stack[:, 1:-(filter_sizes[1] // 2) - 1, :, :])
            # self.Y = h_stack
        x_h_shape = x_h.get_shape().as_list()

        # Now creating fully connected layer
        combined_layer1 = self.get_conv_layer(x_h,
                                              x_h_shape,
                                              filter_size=(1, 1),
                                              padding=None,
                                              out_channels=features,
                                              stride=strides,
                                              masking=None,
                                              activation=self.activation,
                                              scope="combined_layer1")
        combined_layer1_shape = combined_layer1.get_shape().as_list()

        if q_levels is not None:
            out_dim = input_dims[3] * q_levels
        else:
            out_dim = input_dims[3]

        combined_layer2 = self.get_conv_layer(combined_layer1,
                                              combined_layer1_shape,
                                              filter_size=(1, 1),
                                              padding=None,
                                              out_channels=out_dim,
                                              stride=strides,
                                              masking=None,
                                              activation=None,
                                              scope="combined_layer2")

        if q_levels is None:
            self.Y = combined_layer2
        else:
            self.Y = tf.reshape(combined_layer2, [input_dims[0], input_dims[
                                1], input_dims[2], input_dims[3], q_levels])
        # self.Y = combined_layer2

    def get_conv_layer(self,
                       inputs,
                       input_dims,
                       filter_size,
                       padding=None,
                       out_channels=32,
                       stride=[1, 1],
                       masking=None,
                       activation=None,
                       weight_initializer=tf.contrib.layers.xavier_initializer(),
                       bias_initializer=tf.zeros_initializer,
                       weight_regularizer=None,
                       bias_regularizer=None,
                       scope="i_stack"):
        with tf.variable_scope(scope):
            if padding is not None:
                pad_vertical_up = tf.zeros([input_dims[0], padding[0], input_dims[2], input_dims[3]])
                pad_vertical_bottom = tf.zeros(
                    [input_dims[0], padding[0], input_dims[2], input_dims[3]])

                inputs = tf.concat(1, absolute_import[pad_vertical_up, inputs, pad_vertical_bottom])

                new_input_shape = inputs.get_shape().as_list()

                pad_horizontal_up = tf.zeros([new_input_shape[0], new_input_shape[
                                             1], padding[1], new_input_shape[3]])
                pad_horizontal_bottom = tf.zeros([new_input_shape[0], new_input_shape[
                                                 1], padding[1], new_input_shape[3]])

                inputs = tf.concat(
                    2, [pad_horizontal_up, inputs, pad_horizontal_bottom])
            filter_size = filter_size + (input_dims[3], out_channels)

            filter_weights = tf.get_variable(
                "weights", filter_size, tf.float32, weight_initializer, weight_regularizer)
            if masking is not None:

                mask = np.ones(filter_size)
                mask[:, -1, filter_size[1] // 2 + 1:] = 0.0
                if masking.lower() == "a":
                    mask[:, -1, filter_size[1] // 2] = 0.0
                filter_weights = filter_weights * \
                    tf.constant(mask, dtype=tf.float32)

            output = tf.nn.conv2d(inputs, filter_weights, strides=[
                                  1, stride[0], stride[1], 1], padding="VALID")

            if bias_initializer is not None:
                bias = tf.get_variable(
                    "biases", [out_channels, ], tf.float32, bias_initializer, bias_regularizer)
                output = tf.nn.bias_add(output, bias)

            if activation is not None:
                output = activation(output)

            return output

    def figure_out_act(self, activation):
        """
        Sets the global activation for the PixelCNN network

        Args:
            activation: type of activation as a string
        Returns:
            An activation function
        """
        if activation == "relu":
            return tf.nn.relu
        elif activation == "tanh":
            return tf.tanh
        elif activation == "softplus":
            return tf.nn.softplus

        return tf.nn.relu

    def get_output(self):
        """
        Return the model generated images

        Args:
            None
        Returns:
            Output image tensor
        """
        return self.Y


if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    import tensorflow as tf
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                                  tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={
          x: mnist.test.images, y_: mnist.test.labels}))
