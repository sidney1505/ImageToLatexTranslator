import code
import tensorflow as tf
from Decoder import Decoder

class StackedEncBahdanauDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self):
        cells = []
        for i in range(self.model.number_of_layers):
            cells.append(tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size))
        initial_states = tuple(self.model.input_summary)
        rnncell = tf.contrib.rnn.MultiRNNCell(cells)
        # code.interact(local=dict(globals(), **locals()))
        attention = tf.contrib.seq2seq.BahdanauAttention(self.model.decoder_size, \
        	self.model.refined_features)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(rnncell, attention)
        self.initial_state = attention_cell.zero_state(batch_size=self.batchsize, \
        	dtype=tf.float32).clone(cell_state=initial_states)
        return attention_cell
