import code
import tensorflow as tf
from Decoder import Decoder

class BahdanauDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self):
        rnncell = tf.contrib.rnn.BasicLSTMCell(self.model.input_summary[0].shape[-1].value)
        attention = tf.contrib.seq2seq.BahdanauAttention(self.model.decoder_size, \
            self.model.refined_features)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper( \
            rnncell, \
            attention, \
            alignment_history= not self.model.beamsearch and self.model.only_inference)
        self.initial_state = attention_cell.zero_state(batch_size=self.batchsize, \
            dtype=tf.float32).clone(cell_state=self.model.input_summary)
        # code.interact(local=dict(globals(), **locals()))
        return attention_cell
