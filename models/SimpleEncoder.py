import code
import tensorflow as tf
import tflearn

def createGraph(model,features):
    model.encoder = "simpleEncoder"
    with tf.variable_scope(model.encoder, reuse=None):
        shape = tf.shape(features)
        batchsize = shape[0]
        num_features = features.shape[3].value
        output = tf.reshape(features,[batchsize,shape[1]*shape[2],num_features])
        state_hidden = tflearn.layers.conv.global_avg_pool(features)
        state = tflearn.layers.conv.global_max_pool(features)
    return output, (state_hidden, state)