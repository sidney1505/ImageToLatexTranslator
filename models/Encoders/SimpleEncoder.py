import tensorflow as tf
import tflearn
from Encoder import Encoder
import code

class SimpleEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope(self.model.encoder, reuse=None):
            features = self.model.features
            shape = tf.shape(features)
            batchsize = shape[0]
            num_features = features.shape[3].value
            w = tf.get_variable('weight', [num_features, self.model.encoder_size], \
                tf.float32, \
                tf.random_normal_initializer())
            b = tf.get_variable('bias', [self.model.encoder_size], tf.float32, \
                    tf.random_normal_initializer())
            features = tf.tensordot(features,w,[[-1],[0]]) + b
            self.model.refined_features = tf.reshape(features, \
                [batchsize, \
                shape[1]*shape[2], \
                self.model.encoder_size])
            c = tflearn.layers.conv.global_avg_pool(features)
            h = tflearn.layers.conv.global_max_pool(features)
            self.model.input_summary = tf.contrib.rnn.LSTMStateTuple(c,h)