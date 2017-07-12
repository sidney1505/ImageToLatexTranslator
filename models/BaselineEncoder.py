import code
import os
import shutil
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def createBaselineEncoder(model,features):
    shape = tf.shape(features)
    batchsize = shape[0]
    num_features = features.shape[3].value
    features = tf.reshape(features,[batchsize,shape[1]*shape[2],num_features])
    rnncell_fw = tf.contrib.rnn.BasicLSTMCell(1024) # TODO choose parameter
    fw_state = rnncell_fw.zero_state(batch_size=batchsize, dtype=tf.float32)
    rnncell_bw = tf.contrib.rnn.BasicLSTMCell(1024) # TODO choose parameter
    bw_state = rnncell_bw.zero_state(batch_size=batchsize, dtype=tf.float32)
    output, state = \
        tf.nn.bidirectional_dynamic_rnn(rnncell_fw, \
        rnncell_bw, features, initial_state_fw=fw_state, initial_state_bw=bw_state)
    state_fw, state_bw = state
    state_fw_hidden, state_fw = state_fw
    state_bw_hidden, state_bw = state_bw
    #code.interact(local=dict(globals(), **locals())) 
    state_hidden = tf.concat([state_fw_hidden,state_bw_hidden],1)
    state = tf.concat([state_fw,state_bw],1)
    return state_hidden, state