import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder
import code

class StackedQuadroEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)

    def createGraph(self):
        self.model.number_of_layers = 2
        with tf.variable_scope(self.model.encoder, reuse=None):
            self.channels = self.model.encoder_size / 4
            self.encodeRowsBidirectionalStacked()
            self.encodeColsBidirectionalStacked()
            refined_features = tf.concat([self.refined_rows, self.refined_cols], -1)
            batch = tf.shape(refined_features)[0]
            height = tf.shape(refined_features)[1]
            width = tf.shape(refined_features)[2]
            self.model.refined_features = tf.reshape(refined_features, [batch, height * \
                width, self.model.encoder_size])
            s = []
            for layer in range(self.model.number_of_layers):                
                c = tf.concat([self.row_summary[layer][0], self.col_summary[layer][0]], -1)
                h = tf.concat([self.row_summary[layer][1], self.col_summary[layer][1]], -1)
                s.append(tf.contrib.rnn.LSTMStateTuple(c,h))
            self.model.input_summary = s
            # code.interact(local=dict(globals(), **locals()))