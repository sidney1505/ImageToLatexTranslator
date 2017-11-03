import code
import tensorflow as tf
from Decoder import Decoder

class BahdanauDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self):
        rnncell = tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size)
        # code.interact(local=dict(globals(), **locals()))
        attention = tf.contrib.seq2seq.BahdanauAttention(self.model.decoder_size, \
        	self.model.refined_features)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(rnncell, attention)
        self.initial_state = attention_cell.zero_state(batch_size=self.batchsize, \
        	dtype=tf.float32).clone(cell_state=self.model.input_summary)
        return attention_cell
