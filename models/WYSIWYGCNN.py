import code
import os
import shutil
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def createCNNModel(self, features):
    features = tf.nn.conv2d(features, self.weights['wconv1'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, self.weights['bconv1'])
    features = max_pool_2d(features, 2, strides=2)
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, self.weights['wconv2'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, self.weights['bconv2'])
    features = max_pool_2d(features, 2, strides=2)
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, self.weights['wconv3'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, self.weights['bconv3'])
    features = tflearn.layers.normalization.batch_normalization(features) #same as torch?
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, self.weights['wconv4'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, self.weights['bconv4'])
    features = max_pool_2d(features, [2,1], strides=[2,1])
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, self.weights['wconv5'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, self.weights['bconv5'])
    features = tflearn.layers.normalization.batch_normalization(features) #same as torch?
    features = max_pool_2d(features, [1,2], strides=[1,2])
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, self.weights['wconv6'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, self.weights['bconv6'])
    features = tflearn.layers.normalization.batch_normalization(features)
    features = tf.nn.relu(features)
    return features