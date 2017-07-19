import code
import os
import shutil
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def createDecoderLSTM(model, network):
    with tf.variable_scope("wysiwygDecoder", reuse=None):
        shape = tf.Print(tf.shape(network),[tf.shape(network)], \
            "dynamic network shape from decoder input!!!!!!",1)
        dim_beta = 50 # hyperparameter!!!
        dim_h = 512 # hyperparameter!!!
        batchsize = shape[0] # tf.shape(network)[0] # besserer weg???
        num_features = network.shape[3].value
        dim_o = num_features + dim_h
        with tf.variable_scope("initialiseVariables", reuse=None):
            # the used rnncell
            rnncell = tf.contrib.rnn.LSTMCell(256, state_is_tuple=False) #size is hyperparamter!!!
            # used weight variables
            # size is hyperparamter!!!
            beta = tf.get_variable('beta',[dim_beta], tf.float32, \
                tf.random_normal_initializer())
            # dim_beta x dim_h
            w1 = tf.get_variable('w1',[dim_beta, dim_h], tf.float32, \
                tf.random_normal_initializer())
            # num_features x dim_beta
            w2 = tf.get_variable('w2',[num_features, dim_beta], tf.float32, \
                tf.random_normal_initializer())
            # (num_features + dim_h) (x hyperparam) ??? (= dim_o) ???
            wc = tf.get_variable('wc',[dim_o], tf.float32, \
                tf.random_normal_initializer())
            # num_classes x dim_o
            wout = tf.get_variable('wout',[model.num_classes, dim_o], tf.float32, \
                tf.random_normal_initializer())
            # initial states, ebenfalls Variablen???
            h0 = tf.random_normal([batchsize,dim_h]) # 
            y0 = tf.random_normal([batchsize,model.num_classes]) #
            o0 = tf.random_normal([batchsize,dim_o]) #
            # brings network in shape for element wise multiplikation
            network_ = tf.expand_dims(network,4)

        with tf.variable_scope("decoderLSTM", reuse=None):
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
                # put rows and columns in front in order to sum over them
                c = tf.transpose(c, perm=[1,2,0,3]) 
                c = tf.foldl(lambda a, x: a + x, c)
                c = tf.foldl(lambda a, x: a + x, c) # batchsize x num_features
                oput,h = rnncell.__call__(tf.concat([y,o],1), h)
                o = tf.tanh(wc * tf.concat([h,c],1)) # batchsize x dim_o
                # batch_size x num_classes # zeilenweise normieren!!!
                y = tf.nn.softmax(tf.tensordot(o,wout,[[1],[1]])) 
                l = l.write(t, y)
                t = tf.add(t, 1)
                return [t, network, l, h, y, o, beta, network_, w1, w2]
            t, network, l, h, y, o, beta, network_, w1, \
                w2 = tf.while_loop(while_condition, body, params)
            l = l.stack()
        l = tf.transpose(l, [1,0,2])
        #l = tf.Print(l,[tf.shape(l)],"after decoder: ")
        return l