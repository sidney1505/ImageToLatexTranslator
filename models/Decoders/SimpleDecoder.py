import code
import tensorflow as tf
from Decoder import Decoder

class SimpleDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size)

