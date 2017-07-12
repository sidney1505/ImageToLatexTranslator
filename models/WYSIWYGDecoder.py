import code
import os
import shutil
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def createDecoderLSTM(model, network):
    shape = tf.Print(tf.shape(network),[tf.shape(network)], \
        "dynamic network shape from decoder input!!!!!!",1)
    dim_beta = 50 # hyperparameter!!!
    dim_h = 512 # hyperparameter!!!
    batchsize = shape[0] # tf.shape(network)[0] # besserer weg???
    num_features = network.shape[3].value
    dim_o = num_features + dim_h
    # the used rnncell
    rnncell = tf.contrib.rnn.LSTMCell(256, state_is_tuple=False, reuse=None) #size is hyperparamter!!!
    # used weight variables
    beta = tf.Variable(tf.random_normal([dim_beta])) # size is hyperparamter!!!
    w1 = tf.Variable(tf.random_normal([dim_beta, dim_h])) # dim_beta x dim_h
    w2 = tf.Variable(tf.random_normal([num_features, dim_beta])) # num_features x dim_beta
    wc = tf.Variable(tf.random_normal([dim_o])) # (num_features + dim_h) (x hyperparam) ??? (= dim_o) ???
    wout = tf.Variable(tf.random_normal([model.num_classes, dim_o])) # num_classes x dim_o
    # initial states, ebenfalls Variablen???
    h0 = tf.zeros([batchsize,dim_h]) # 
    y0 = tf.zeros([batchsize,model.num_classes]) #
    o0 = tf.zeros([batchsize,dim_o]) #
    # brings network in shape for element wise multiplikation
    network_ = tf.expand_dims(network,4)
    # create necessaties for the while-loop
    l = tf.TensorArray(dtype=tf.float32, size=model.max_num_tokens)
    params = [tf.constant(0), network, l, h0, y0, o0, beta, network_, w1, w2]
    tmax = tf.constant(model.max_num_tokens)
    #tmax = tf.Print(tmax,[tmax],"tmax =============================")
    def while_condition(t, network, l, h, y, o, beta, network_, w1, w2):
        #t = tf.Print(t,[t],"twhile =============================")
        return tf.less(t, tmax) # token embedding??
    def body(t, network, l, h, y, o, beta, network_, w1, w2):
        hdotw1 = tf.tensordot(h,w1,[[1],[1]])
        hdotw1 = tf.expand_dims(hdotw1,1)
        hdotw1 = tf.expand_dims(hdotw1,1)
        hdotw1 = tf.expand_dims(hdotw1,1)
        e = tf.tensordot(beta,tf.tanh(hdotw1 + w2 * network_),[[0],[4]])
        alpha = tf.nn.softmax(e, dim=1)
        c = alpha * network
        c = tf.transpose(c, perm=[1,2,0,3]) # put rows and columns in front in order to sum over them
        c = tf.foldl(lambda a, x: a + x, c)
        c = tf.foldl(lambda a, x: a + x, c) # batchsize x num_features
        oput,h = rnncell.__call__(tf.concat([y,o],1), h)
        o = tf.tanh(wc * tf.concat([h,c],1)) # batchsize x dim_o
        y = tf.nn.softmax(tf.tensordot(o,wout,[[1],[1]])) # batch_size x num_classes # zeilenweise normieren!!!
        l = l.write(t, y)
        t = tf.add(t, 1)
        return [t, network, l, h, y, o, beta, network_, w1, w2]
    t, network, l, h, y, o, beta, network_, w1, \
        w2 = tf.while_loop(while_condition, body, params)
    #t = tf.Print(t,[t],"ttttttttttttttttttttttttttttend =============================",5)
    l = l.stack()
    l = tf.transpose(l, [1,0,2])
    #l = tf.Print(l,[tf.shape(l)],"after decoder: ")
    return l