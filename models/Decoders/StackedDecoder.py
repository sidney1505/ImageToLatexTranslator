import code
import tensorflow as tf
from Decoder import Decoder

class StackedDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self):
    	self.initial_state = self.model.input_summary
    	number_of_layers = 2
    	def lstm_cell():
			return tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size)
		stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in \
			range(number_of_layers)])
        return stacked_lstm

