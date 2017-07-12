import code
import os
import shutil
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def createCNNModel(model, inputpx):
    model.weights.update({
        # 5x5 conv, 1 input, 32 outputs
        'wconv1': tf.Variable(tf.random_normal([3, 3, 1, 64]), name='wconv1'),
        'bconv1': tf.Variable(tf.random_normal([64]), name='bconv1'),
        # 5x5 conv, 32 inputs, 64 outputs
        'wconv2': tf.Variable(tf.random_normal([3, 3, 64, 128]), name='wconv2'),
        'bconv2': tf.Variable(tf.random_normal([128]), name='bconv2'),
        # 5x5 conv, 1 input, 32 outputs
        'wconv3': tf.Variable(tf.random_normal([3, 3, 128, 256]), name='wconv3'),
        'bconv3': tf.Variable(tf.random_normal([256]), name='bconv3'),
        # 5x5 conv, 32 inputs, 64 outputs
        'wconv4': tf.Variable(tf.random_normal([3, 3, 256, 256]), name='wconv4'),
        'bconv4': tf.Variable(tf.random_normal([256]), name='bconv4'),
        # 5x5 conv, 1 input, 32 outputs
        'wconv5': tf.Variable(tf.random_normal([3, 3, 256, 512]), name='wconv5'),
        'bconv5': tf.Variable(tf.random_normal([512]), name='bconv5'),
        # 5x5 conv, 32 inputs, 64 outputs
        'wconv6': tf.Variable(tf.random_normal([3, 3, 512, 512]), name='wconv6'),
        'bconv6': tf.Variable(tf.random_normal([512]), name='bconv6'),
    })
    features = tf.nn.conv2d(inputpx, model.weights['wconv1'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, model.weights['bconv1'])
    features = max_pool_2d(features, 2, strides=2)
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, model.weights['wconv2'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, model.weights['bconv2'])
    features = max_pool_2d(features, 2, strides=2)
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, model.weights['wconv3'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, model.weights['bconv3'])
    features = tflearn.layers.normalization.batch_normalization(features) #same as torch?
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, model.weights['wconv4'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, model.weights['bconv4'])
    features = max_pool_2d(features, [2,1], strides=[2,1])
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, model.weights['wconv5'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, model.weights['bconv5'])
    features = tflearn.layers.normalization.batch_normalization(features) #same as torch?
    features = max_pool_2d(features, [1,2], strides=[1,2])
    features = tf.nn.relu(features)

    features = tf.nn.conv2d(features, model.weights['wconv6'], strides=[1,1,1,1], \
        padding='SAME')
    features = tf.nn.bias_add(features, model.weights['bconv6'])
    features = tflearn.layers.normalization.batch_normalization(features)
    features = tf.nn.relu(features)
    return features