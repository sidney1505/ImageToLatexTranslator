import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder

class RowcolEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)

    def createGraph(self):
        self.channels = self.model.encoder_size / 2
        self.encodeRows()
        self.encodeCols()
        refined_features = tf.concat([self.refined_rows, self.refined_cols], -1)
        batch = tf.shape(refined_features)[0]
        height = tf.shape(refined_features)[1]
        width = tf.shape(refined_features)[2]
        self.model.refined_features = tf.reshape(refined_features, [batch, height * \
            width, self.model.encoder_size])
        c = tf.concat([self.row_summary[0], self.col_summary[0]], -1)
        h = tf.concat([self.row_summary[1], self.col_summary[1]], -1)
        self.model.input_summary = tf.contrib.rnn.LSTMStateTuple(c,h)