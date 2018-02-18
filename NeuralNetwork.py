from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.python.ops import io_ops
import time as t

start = t.time()

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def init_weights(shape):
    """
    initializes weights and biases to passed in shape
    :param shape: array of dims
    :return: weights and biases
    """
    w = tf.truncated_normal(shape, stddev=0.1)
    b = tf.constant(0.1, shape=[shape[-1]])
    return tf.Variable(w), tf.Variable(b)

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
W1, b1 = init_weights([20, 9, 1, 64])
conv1 = tf.nn.conv2d(signal, W1, strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(conv1 + b1)
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='SAME')

W2, b2 = init_weights([10, 4, 64, 64])
conv2 = tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME')
relu2 = tf.nn.relu(conv2 + b2)

out_height = relu2.get_shape()[1]
out_width = relu2.get_shape()[2]

# fully-connected layer
W3, b3 = init_weights([out_height*out_width*64, label_count])
flatten = tf.reshape(relu2, [-1, out_height*out_width*64])
z = tf.matmul(flatten, W3) + b3


'''
# cost function
sm = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=z)
cross_entropy = tf.reduce_mean(sm)

# training optimizer
train = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

# accuracy
correct = tf.equal(tf.argmax(z, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# run graph

sess.run(tf.global_variables_initializer())
for i in range(5000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0
        })
        print('step %d\ttraining accuracy: %g' % (i, train_accuracy))
    train.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print('test accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
}))

end = t.time()

print('runtime: %f' % (end-start))
'''
