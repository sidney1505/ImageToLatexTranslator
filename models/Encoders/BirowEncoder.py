import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder

class BirowEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope(self.model.encoder, reuse=None):
            self.channels = self.model.encoder_size / 2
            self.encodeRowsBidirectional()
            refined_features = self.refined_rows
            batch = tf.shape(refined_features)[0]
            height = tf.shape(refined_features)[1]
            width = tf.shape(refined_features)[2]
            self.model.refined_features = tf.reshape(refined_features, [batch, height * \
                width, self.model.encoder_size])
            self.model.input_summary = self.row_summary