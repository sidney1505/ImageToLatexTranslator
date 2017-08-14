import numpy as np
import tensorflow as tf
import argparse
import os


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

def createCNNModel(self, input_vars):
    image, label = input_vars

    def conv(name, l, channel, stride):
        return Conv2D(name, l, channel, 3, stride=stride,
                      nl=tf.identity, use_bias=False,
                      W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))
    def add_layer(name, l):
        shape = l.get_shape().as_list()
        in_channel = shape[3]
        with tf.variable_scope(name) as scope:
            c = BatchNorm('bn1', l)
            c = tf.nn.relu(c)
            c = conv('conv1', c, self.growthRate, 1)
            l = tf.concat([c, l], 3)
        return l

    def add_transition(name, l):
        shape = l.get_shape().as_list()
        in_channel = shape[3]
        with tf.variable_scope(name) as scope:
            l = BatchNorm('bn1', l)
            l = tf.nn.relu(l)
            l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
            l = AvgPooling('pool', l, 2)
        return l


    def dense_net(name):
        l = conv('conv0', image, 16, 1)
        with tf.variable_scope('block1') as scope:

            for i in range(self.N):
                l = add_layer('dense_layer.{}'.format(i), l)
            l = add_transition('transition1', l)

        with tf.variable_scope('block2') as scope:

            for i in range(self.N):
                l = add_layer('dense_layer.{}'.format(i), l)
            l = add_transition('transition2', l)

        with tf.variable_scope('block3') as scope:

            for i in range(self.N):
                l = add_layer('dense_layer.{}'.format(i), l)
        l = BatchNorm('bnlast', l)
        l = tf.nn.relu(l)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)

        return logits

    logits = dense_net("dense_net")

    return logits