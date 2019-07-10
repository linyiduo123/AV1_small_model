import tensorflow as tf
import numpy as np


def conv2d(tensor, shape, name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        conv_w = tf.get_variable("conv_w", shape, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/(shape[0]*shape[1]*shape[2]))))
        conv_b = tf.get_variable("conv_b", [shape[-1]], initializer=tf.constant_initializer(0))
        # for weight decay using
        tf.add_to_collection(tf.GraphKeys.WEIGHTS,tf.contrib.layers.l2_regularizer(1.)(conv_w))
        convResult = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)

    return convResult


def model(input_tensor):
    #with tf.device("/gpu:0"):
        # layer 1
        tensor = tf.nn.relu(conv2d(input_tensor, [3,3,1,64], "layer_1"))

        #for i in range(5):
        tensor = tf.nn.relu(conv2d(tensor, [3,3,64,64], "layer_%02d"%(2)))
        tensor = tf.nn.relu(conv2d(tensor, [3,3,64,48], "layer_%02d"%(3)))
        tensor = tf.nn.relu(conv2d(tensor, [3,3,48,32], "layer_%02d"%(4)))
        tensor = tf.nn.relu(conv2d(tensor, [3,3,32,16], "layer_%02d"%(5)))
        tensor = tf.nn.relu(conv2d(tensor, [3,3,16,64], "layer_%02d"%(6)))

        # the last layer
        tensor = conv2d(tensor, [3,3,64,1], "last_layer")

        tensor = tf.add(tensor, input_tensor)
        return tensor

