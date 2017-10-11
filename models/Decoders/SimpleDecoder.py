import code
import tensorflow as tf
from Decoder import Decoder

class SimpleDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self, statesize, features):
        return tf.contrib.rnn.BasicLSTMCell(statesize)

