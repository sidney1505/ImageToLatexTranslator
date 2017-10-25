import code
import tensorflow as tf
from Decoder import Decoder

class LuongDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self):
        rnncell = tf.contrib.rnn.BasicLSTMCell(self.statesize)
        attention = tf.contrib.seq2seq.LuongAttention(self.model.decoder_size, \
        	self.model.refined_features)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(rnncell, attention)
        return rnncell
