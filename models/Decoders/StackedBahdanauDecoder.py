import code
import tensorflow as tf
from Decoder import Decoder

class StackedBahdanauDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self):
        number_of_layers = 2
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size)
        rnncell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in \
            range(number_of_layers)])
        # code.interact(local=dict(globals(), **locals()))
        attention = tf.contrib.seq2seq.BahdanauAttention(self.model.decoder_size, \
        	self.model.refined_features)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(rnncell, attention)
        self.initial_state = attention_cell.zero_state(batch_size=self.batchsize, \
        	dtype=tf.float32).clone(cell_state=self.model.input_summary)
        return attention_cell
