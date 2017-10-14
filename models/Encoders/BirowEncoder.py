import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder

class BirowEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope(self.model.encoder, reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            num_features = self.model.features.shape[3].value
            #code.interact(local=dict(globals(), **locals()))
            features = tf.reshape(self.model.features,[batchsize,shape[1]*shape[2], \
                num_features])
            rnncell_fw = tf.contrib.rnn.BasicLSTMCell(1024) # TODO choose parameter
            fw_state = rnncell_fw.zero_state(batch_size=batchsize, dtype=tf.float32)
            rnncell_bw = tf.contrib.rnn.BasicLSTMCell(1024) # TODO choose parameter
            bw_state = rnncell_bw.zero_state(batch_size=batchsize, dtype=tf.float32)
            output, state = \
                tf.nn.bidirectional_dynamic_rnn(rnncell_fw, \
                rnncell_bw, features, initial_state_fw=fw_state, initial_state_bw=bw_state,
                parallel_iterations=1)
            state_fw, state_bw = state
            state_fw_hidden, state_fw = state_fw
            state_bw_hidden, state_bw = state_bw
            state_hidden = tf.concat([state_fw_hidden,state_bw_hidden],1)
            state = tf.concat([state_fw,state_bw],1)
            output = tf.concat([output[0],output[1]],2)
        return output, (state_hidden, state)