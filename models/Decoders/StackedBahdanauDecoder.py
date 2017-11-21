import code
import tensorflow as tf
from Decoder import Decoder

class StackedBahdanauDecoder(Decoder):
    def __init__(self, model):
        Decoder.__init__(self, model)

    def createDecoderCell(self):
        number_of_layers = 4
        cells = [tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size)]
        initial_states = [self.model.input_summary]
        for i in range(number_of_layers - 1):
            cells.append(tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size))
            last = initial_states[-1]
            c, h = last
            wc = tf.get_variable('weight_c' + str(i), [c.shape[-1], c.shape[-1]], \
                tf.float32, tf.random_normal_initializer())
            bc = tf.get_variable('bias_c' + str(i), [c.shape[-1]], tf.float32, \
                tf.random_normal_initializer())
            cn = tf.tensordot(c,wc,[[-1],[0]]) + bc
            wh = tf.get_variable('weight_h' + str(i), [h.shape[-1], h.shape[-1]], \
                tf.float32, tf.random_normal_initializer())
            bh = tf.get_variable('bias_h' + str(i), [h.shape[-1]], tf.float32, \
                tf.random_normal_initializer())
            hn = tf.tensordot(h,wh,[[-1],[0]]) + bh
            current = tf.contrib.rnn.LSTMStateTuple(cn,hn)
            initial_states.append(current)
        initial_states = tuple(initial_states)
        rnncell = tf.contrib.rnn.MultiRNNCell(cells)
        # code.interact(local=dict(globals(), **locals()))
        attention = tf.contrib.seq2seq.BahdanauAttention(self.model.decoder_size, \
        	self.model.refined_features)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(rnncell, attention)
        self.initial_state = attention_cell.zero_state(batch_size=self.batchsize, \
        	dtype=tf.float32).clone(cell_state=initial_states)
        return attention_cell
