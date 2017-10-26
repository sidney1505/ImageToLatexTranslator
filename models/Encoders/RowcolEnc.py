import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder

class RowcolEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope(self.model.encoder, reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            height = shape[1]
            width = shape[2]
            num_features = self.model.features.shape[3].value
            #
            channels = self.model.encoder_size / 2
            # rows
            rowrnncell = tf.contrib.rnn.BasicLSTMCell(channels)
            rowfeatures = tf.reshape(self.model.features, [batchsize * height, width, \
                num_features])
            # create the postional row embeddings
            initial_state_rows = tf.range(height, dtype=tf.float32)
            initial_state_rows = tf.tile(initial_state_rows, channels)
            initial_state_rows = tf.transpose(initial_state_rows)
            initial_state_rows = tf.tile(initial_state_rows, batchsize)
            initial_state_rows = tf.reshape(initial_state_rows, [batchsize*height, channels)
            initial_state_rows = tf.contrib.rnn.LSTMStateTuple((initial_state_rows, \
                initial_state_rows))
            # run the actual row encoding lstm
            rowfeatures, rowstates = tf.nn.dynamic_rnn(rowrnncell, rowfeatures, \
                initial_state=initial_state_rows)
            #
            self.model.refined_features = tf.reshape(rowfeatures, [batchsize,height,width,channels])
            #
            rowstates = tf.reshape(rowstates, [batchsize, height, channels])
            rowstaternncell = tf.contrib.rnn.BasicLSTMCell(channels)
            _, self.model.input_summary = tf.nn.dynamic_rnn(rowstaternncell, rowstates)