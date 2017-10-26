import code
import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder

class MonorowEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope(self.model.encoder, reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            height = shape[1]
            width = shape[2]
            num_features = self.model.features.shape[3].value
            #
            channels = self.model.encoder_size
            # rows
            rowrnncell = tf.contrib.rnn.BasicLSTMCell(channels)
            rowfeatures = tf.reshape(self.model.features, [batchsize * height, width, \
                num_features])
            # create the postional row embeddings
            
            initial_state_rows = tf.range(height)
            initial_state_rows = tf.expand_dims(initial_state_rows, 0)
            initial_state_rows = tf.tile(initial_state_rows, [channels, 1])

            initial_state_rows = tf.transpose(initial_state_rows) # WxC
            initial_state_rows = tf.expand_dims(initial_state_rows, 0)# 1xWxC
            initial_state_rows = tf.tile(initial_state_rows, [batchsize, 1, 1]) # BxWxC
                                    
            initial_state_rows = tf.reshape(initial_state_rows, [batchsize*height, channels])
            initial_state_rows = tf.cast(initial_state_rows, tf.float32)
            initial_state_rows = tf.contrib.rnn.LSTMStateTuple(initial_state_rows, \
                initial_state_rows)
            # run the actual row encoding lstm
            #code.interact(local=dict(globals(), **locals()))
            rowfeatures, rowstates = tf.nn.dynamic_rnn(rowrnncell, rowfeatures, \
                initial_state=initial_state_rows)

        with tf.variable_scope(self.model.encoder + '2', reuse=None):
            #
            self.model.refined_features = tf.reshape(rowfeatures, [batchsize,height,width,\
                channels]) # (20, 4, 48, 2048)            
            #
            rowstates_c = tf.reshape(rowstates[0], [batchsize, height, channels])
            rowstates_h = tf.reshape(rowstates[1], [batchsize, height, channels])
            rowstates = tf.concat([rowstates_c, rowstates_h],-1)
            #
            w1 = tf.get_variable('weight', [2 * channels, channels], tf.float32, \
                    tf.random_normal_initializer())
            b1 = tf.get_variable('bias', [channels], tf.float32, \
                    tf.random_normal_initializer())
            rowstates = tf.tensordot(rowstates,w1,[[-1],[0]]) + b1
            rowstates = tf.nn.relu(rowstates)
            rowstates = tf.nn.dropout(rowstates,self.model.keep_prob)
            # 
            rowstaternncell = tf.contrib.rnn.BasicLSTMCell(channels)
            _, rowstates = tf.nn.dynamic_rnn(rowstaternncell, rowstates, \
                dtype=tf.float32)
            rowstates = tf.reshape(rowstates, [2, batchsize, channels])
            self.model.input_summary = tf.contrib.rnn.LSTMStateTuple(rowstates[0], \
                rowstates[1])