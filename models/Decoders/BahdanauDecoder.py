import code
import tensorflow as tf
from Decoder import Decoder

class BahdanauDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self, statesize, features):
        rnncell = tf.contrib.rnn.BasicLSTMCell(statesize)
        attention = tf.contrib.seq2seq.BahdanauAttention(1024, features) # hyperparam
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(rnncell, attention)
        return rnncell
