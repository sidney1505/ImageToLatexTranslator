import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder

class MonorowEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope(self.model.encoder, reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            num_features = self.model.features.shape[3].value
            features = tf.reshape(self.model.features,[batchsize,shape[1]*shape[2], \
                num_features])
            rnncell = tf.contrib.rnn.BasicLSTMCell(2048) # hyperparamter
            state = rnncell.zero_state(batch_size=batchsize, dtype=tf.float32)
            output, state = tf.nn.dynamic_rnn(rnncell, features, initial_state=state)
        return output, state