import code
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import Decoder

import abc
from abc_base import PluginBase

@PluginBase.register
class SimpleDecoder(Decoder):
    def createDecoderCell(self, statesize):
        return tf.contrib.rnn.BasicLSTMCell(statesize)
