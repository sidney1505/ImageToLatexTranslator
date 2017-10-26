import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder

class MonocolEncoder(Encoder):
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
            # cols
            colrnncell = tf.contrib.rnn.BasicLSTMCell(channels)
            colfeatures = tf.transpose(self.model.features, [0,2,1,3])
            colfeatures = tf.reshape(self.model.features, [batchsize * width, height, \
                num_features])
            # create the postional col embeddings
            
            initial_state_cols = tf.range(width)
            initial_state_cols = tf.expand_dims(initial_state_cols, 0)
            initial_state_cols = tf.tile(initial_state_cols, [channels, 1])

            initial_state_cols = tf.transpose(initial_state_cols) # WxC
            initial_state_cols = tf.expand_dims(initial_state_cols, 0)# 1xWxC
            initial_state_cols = tf.tile(initial_state_cols, [batchsize, 1, 1]) # BxWxC
                                    
            initial_state_cols = tf.reshape(initial_state_cols, [batchsize*width, channels])
            initial_state_cols = tf.cast(initial_state_cols, tf.float32)
            initial_state_cols = tf.contrib.rnn.LSTMStateTuple(initial_state_cols, \
                initial_state_cols)
            # run the actual col encoding lstm
            #code.interact(local=dict(globals(), **locals()))
            colfeatures, colstates = tf.nn.dynamic_rnn(colrnncell, colfeatures, \
                initial_state=initial_state_cols)

        with tf.variable_scope(self.model.encoder + '2', reuse=None):
            #
            colfeatures = tf.reshape(colfeatures, [batchsize,width,height,\
                channels]) # (20, 4, 48, 2048)
            self.model.refined_features = tf.transpose(colfeatures, [0,2,1,3])
            #
            colstates_c = tf.reshape(colstates[0], [batchsize, width, channels])
            colstates_h = tf.reshape(colstates[1], [batchsize, width, channels])
            colstates = tf.concat([colstates_c, colstates_h],-1)
            #
            w1 = tf.get_variable('weight', [2 * channels, channels], tf.float32, \
                    tf.random_normal_initializer())
            b1 = tf.get_variable('bias', [channels], tf.float32, \
                    tf.random_normal_initializer())
            colstates = tf.tensordot(colstates,w1,[[-1],[0]]) + b1
            colstates = tf.nn.relu(colstates)
            colstates = tf.nn.dropout(colstates,self.model.keep_prob)
            # 
            colstaternncell = tf.contrib.rnn.BasicLSTMCell(channels)
            _, colstates = tf.nn.dynamic_rnn(colstaternncell, colstates, \
                dtype=tf.float32)
            colstates = tf.reshape(colstates, [2, batchsize, channels])
            self.model.input_summary = tf.contrib.rnn.LSTMStateTuple(colstates[0], \
                colstates[1])