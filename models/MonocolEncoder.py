import code
import tensorflow as tf

def createGraph(model,features):
    model.encoder = "basicEncoder"
    with tf.variable_scope(model.encoder, reuse=None):
        shape = tf.shape(features)
        batchsize = shape[0]
        num_features = features.shape[3].value
        features = tf.transpose(features,[0,2,1,3])
        features = tf.reshape(features,[batchsize,shape[1]*shape[2],num_features])
        rnncell = tf.contrib.rnn.BasicLSTMCell(2048) # hyperparamter
        state = rnncell.zero_state(batch_size=batchsize, dtype=tf.float32)
        output, state = tf.nn.dynamic_rnn(rnncell, features, initial_state=state)
    return output, state