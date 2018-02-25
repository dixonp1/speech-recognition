import tensorflow as tf
from post_processing import recommend_word

class model:

    def __init__(self, sig_features, num_classes):
        self.build_model(sig_features, num_classes)

    def _init_weights(self, shape):
        """
        initializes weights and biases to passed in shape
        :param shape: array of dims
        :return: weights and biases
        """
        w = tf.truncated_normal(shape, stddev=0.1)
        b = tf.constant(0.1, shape=[shape[-1]])
        return tf.Variable(w), tf.Variable(b)

    def build_model(self, sig_features, num_classes):
        '''

            width = frequency domain
            height = time domain
            1st conv
                kernel width = 9
                kernel height = 20 (2/3 in time domain)
                feature maps = 64
                stride = 1
            relu
            max pool  (reduces variability, noise)
                width = 3
                height = 1
            2nd conv
                kernel width = 4
                kernel height = 10
                feature maps = 32
                stride = 1
            relu
            fully-connected
                hidden units = 128
            softmax

        '''
        _, h, w = sig_features.get_shape()
        signal = tf.reshape(sig_features, [-1, 99, 40, 1])

        W1, b1 = self._init_weights([20, 9, 1, 64])
        conv1 = tf.nn.conv2d(signal, W1, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(conv1 + b1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME')

        W2, b2 = self._init_weights([10, 4, 64, 64])
        conv2 = tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(conv2 + b2)

        out_height = relu2.get_shape()[1]
        out_width = relu2.get_shape()[2]

        # fully-connected layer
        W3, b3 = self._init_weights([int(out_height*out_width*64), 128])
        flatten = tf.reshape(relu2, [-1, int(out_height*out_width*64)])
        z1 = tf.matmul(flatten, W3) + b3
        relu3 = tf.nn.relu(z1)

        # softmax
        W4, b4 = self._init_weights([128, num_classes])
        z2 = tf.matmul(relu3, W4) + b4
        # softmax = tf.exp(z2) / tf.reduce_sum(tf.exp(z2), -1)
        #sm = tf.nn.softmax(z2)

        self.softmax = z2

    def forward_prop(self, signal, sess):
        num_frames = tf.shape(signal)[1]
        smoothed_out = []
        for frame in range(num_frames):
            net_out = sess.run(self.softmax)

            #smoothed_out, word, conf = recommend_word(net_out,
             #                                         smoothed_out,
              #                                        frame,
               #                                       w_smooth,
                #                                      w_max,
                 #                                     threshold)

