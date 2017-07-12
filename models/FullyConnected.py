import code
import os
import shutil
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def createFullyConnected(model, features):
    model.weights.update({
        'wfc3': tf.Variable(tf.random_normal([512,1000]), name='wfc3'),
        'bfc3': tf.Variable(tf.random_normal([1000]), name='bfc3'),
        'wfc4': tf.Variable(tf.random_normal([1000,model.num_classes]), \
            name='wfc4'),
        'bfc4': tf.Variable(tf.random_normal([model.num_classes]), name='bfc4'),
    })
    #code.interact(local=dict(globals(), **locals())) 
    features = tflearn.layers.conv.global_avg_pool(features)
    features = tf.tensordot(features,model.weights['wfc3'],[[1],[0]])
    features = features + model.weights['bfc3']
    features = tf.nn.relu(features)
    features = tf.tensordot(features,model.weights['wfc4'],[[1],[0]])
    features = features + model.weights['bfc4']
    return features