import code
import tensorflow as tf
from Decoder import Decoder

class SimplegruDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self):
        return tf.contrib.rnn.GRUCell(self.statesize)
