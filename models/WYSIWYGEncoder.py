import code
import os
import shutil
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def createEncoderLSTM(model, features):
    with tf.variable_scope("wysiwygEncoder", reuse=None):
        batchsize = tf.shape(features)[0]
        features = tf.transpose(features, [1,0,2,3])
        # trainable hidden state??? (postional embedding)
        rnncell_fw = tf.contrib.rnn.LSTMCell(256) # TODO choose parameter
        fw_state = rnncell_fw.zero_state(batch_size=batchsize, dtype=tf.float32)
        rnncell_bw = tf.contrib.rnn.LSTMCell(256) # TODO choose parameter
        bw_state = rnncell_bw.zero_state(batch_size=batchsize, dtype=tf.float32)
        l = tf.TensorArray(dtype=tf.float32, size=tf.shape(features)[0])
        params = [tf.constant(0), features, l, fw_state, bw_state]
        rows = tf.shape(features)[0]
        while_condition = lambda i, features, l, fw_state, bw_state: tf.less(i, rows)
        def body(i, features, l, fw_state, bw_state):
            with tf.variable_scope("encoderLSTM", reuse=None):
                #i = tf.Print(i,[i],"row: ")
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(rnncell_fw, \
                    rnncell_bw, features[i], initial_state_fw=fw_state, \
                    initial_state_bw=bw_state)
                fw_state, bw_state = output_states
                # code.interact(local=locals())
                l = l.write(i, outputs)        
                return [tf.add(i, 1), features, l, fw_state, bw_state]
        i, features, l, fw_state, bw_state = tf.while_loop(while_condition, body, \
            params)
        features = l.stack()
        #code.interact(local=dict(globals(), **locals()))
        #features = tf.Print(features,[tf.shape(features)],"in encoder: ")
        features = tf.transpose(features, perm=[2,0,3,1,4])
        s = tf.shape(features)
        num_features = (features.shape[3] * features.shape[4]).value # other solution???
        #code.interact(local=dict(globals(), **locals()))
        # 
        features = tf.reshape(features, [s[0],s[1],s[2],num_features])
        #features = tf.Print(features,[tf.shape(features)],"after encoder: ")
        return features #