import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder

class QuadroEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope(self.model.encoder, reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            rows = shape[1]
            cols = shape[2]
            num_features = self.model.features.shape[3].value
            col_features = tf.transpose(self.model.features,[0,2,1,3])
            #code.interact(local=dict(globals(), **locals()))
            # rows
            features = tf.reshape(self.model.features,[batchsize,rows*cols,num_features])
            rnncell_fw = tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size / 4) # TODO choose parameter
            fw_state = rnncell_fw.zero_state(batch_size=batchsize, dtype=tf.float32)
            rnncell_bw = tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size / 4) # TODO choose parameter
            bw_state = rnncell_bw.zero_state(batch_size=batchsize, dtype=tf.float32)
            row_output, row_state = tf.nn.bidirectional_dynamic_rnn(rnncell_fw, \
                rnncell_bw, features, initial_state_fw=fw_state, initial_state_bw=bw_state)
            row_state_fw, row_state_bw = row_state
            row_state_fw_hidden, row_state_fw = row_state_fw
            row_state_bw_hidden, row_state_bw = row_state_bw
            # cols
            col_features = tf.reshape(col_features,[batchsize,cols*rows,num_features])
            col_rnncell_fw = tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size / 4) # TODO choose parameter
            col_fw_state = col_rnncell_fw.zero_state(batch_size=batchsize, dtype=tf.float32)
            col_rnncell_bw = tf.contrib.rnn.BasicLSTMCell(self.model.encoder_size / 4) # TODO choose parameter
            col_bw_state = col_rnncell_bw.zero_state(batch_size=batchsize, dtype=tf.float32)
            with tf.variable_scope('col', reuse=None):
                col_output, col_state = tf.nn.bidirectional_dynamic_rnn(col_rnncell_fw, \
                    col_rnncell_bw, col_features, initial_state_fw=col_fw_state, \
                    initial_state_bw=col_bw_state)
            col_output_fw = tf.reshape(col_output[0],[batchsize,cols,rows, \
                col_rnncell_fw.output_size])
            col_output_fw = tf.transpose(col_output_fw,[0,2,1,3])
            col_output_fw = tf.reshape(col_output_fw,[batchsize,rows*cols, \
                col_rnncell_fw.output_size])
            col_output_bw = tf.reshape(col_output[1],[batchsize,cols,rows, \
                col_rnncell_bw.output_size])
            col_output_bw = tf.transpose(col_output_bw,[0,2,1,3])
            col_output_bw = tf.reshape(col_output_bw,[batchsize,rows*cols, \
                col_rnncell_bw.output_size])
            col_state_fw, col_state_bw = col_state
            col_state_fw_hidden, col_state_fw = col_state_fw
            col_state_bw_hidden, col_state_bw = col_state_bw        
            #
            state_hidden = tf.concat([row_state_fw_hidden,row_state_bw_hidden, \
                col_state_fw_hidden, col_state_bw_hidden],1)
            output = tf.concat([row_output[0],row_output[1],col_output_fw,col_output_bw],2)
            state = tf.concat([row_state_fw,row_state_bw,col_state_fw,col_state_bw],1)
            #code.interact(local=dict(globals(), **locals()))
        return output, (state_hidden, state)