import code
import os
import shutil
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def createSimpleAttentionModule(model, features):
    batchsize = tf.shape(features)[0]
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(1024)
    rnn_cell_state = rnn_cell.zero_state(batch_size=batchsize, dtype=tf.float32)
    model.weights.update({
        'watt': tf.Variable(tf.random_normal([2048,512]), name='watt'),
        'batt': tf.Variable(tf.random_normal([512]), name='batt')
    })
    #code.interact(local=dict(globals(), **locals())) 
    outputs = tf.TensorArray(dtype=tf.float32, size=tf.constant(model.max_num_tokens))
    params = [tf.constant(0), features, outputs, rnn_cell_state]
    def while_condition(token,_a,_b,_c):
        return tf.less(token, model.max_num_tokens)
    def body(token, features, outputs, rnn_cell_state):
        #code.interact(local=dict(globals(), **locals()))
        batchsize = tf.shape(features)[0]
        # batch x height x width
        # calculate e
        h1,h2 = rnn_cell_state # richtiger teil???
        # code.interact(local=dict(globals(), **locals()))
        h = tf.concat([h1,h2],1)
        h = tf.tensordot(h,model.weights['watt'],[[1],[0]]) + model.weights['batt']
        h = tf.tanh(h)
        inner_outputs = tf.TensorArray(dtype=tf.float32, \
            size=batchsize)
        inner_params = [tf.constant(0), h, features, inner_outputs]
        def inner_while_condition(batch, h, features, inner_outputs):
            return tf.less(batch, batchsize)
        def inner_body(batch, h, features, inner_outputs):
            inner_outputs = inner_outputs.write(batch, \
                tf.tensordot(h[batch],features[batch],[[0],[2]]))
            return [batch + 1, h, features, inner_outputs]
        _,_,_,inner_outputs = tf.while_loop(inner_while_condition, \
            inner_body,inner_params)
        e = inner_outputs.stack()
        # own softmax
        shape = tf.shape(e)
        alpha = tf.reshape(e,[shape[0],shape[1]*shape[2]])
        alpha = tf.nn.softmax(alpha)
        alpha = tf.reshape(e,[shape[0],shape[1],shape[2]])
        # calculate the context
        inner_outputs = tf.TensorArray(dtype=tf.float32, \
            size=batchsize)
        inner_params = [tf.constant(0), alpha, features, inner_outputs]
        def inner_while_condition(batch, alpha, features, inner_outputs):
            return tf.less(batch, batchsize)
        def inner_body(batch, alpha, features, inner_outputs):
            inner_outputs = inner_outputs.write(batch, \
                tf.tensordot(alpha[batch],features[batch],[[0,1],[0,1]]))
            return [batch + 1, alpha, features, inner_outputs]
        _,_,_,inner_outputs = tf.while_loop(inner_while_condition, \
            inner_body,inner_params)
        context = inner_outputs.stack()
        # apply the lstm
        output, rnn_cell_state = rnn_cell.__call__(context, rnn_cell_state)
        outputs = outputs.write(token, output)
        #code.interact(local=dict(globals(), **locals()))
        return [token + 1, features, outputs, rnn_cell_state]
    _,_,outputs,_= tf.while_loop(while_condition, body, params)
    decoded = outputs.stack() # max_num_tokens x batchsize x num_features
    #code.interact(local=dict(globals(), **locals()))
    decoded = tf.transpose(decoded, [1,0,2]) # batchsize x max_num_tokens x num_features
    model.weights.update({
        'wfc': tf.Variable(tf.random_normal([1024,model.num_classes]), name='wfc'),
        'bfc': tf.Variable(tf.random_normal([model.num_classes]), name='bfc')
    })
    decoded = tf.tensordot(decoded,model.weights['wfc'],[[2],[0]])
    decoded = decoded + model.weights['bfc']
    prediction = tf.nn.softmax(decoded) # batchsize x max_num_tokens x num_classes
    return prediction # batchsize x max_num_tokens x num_classes