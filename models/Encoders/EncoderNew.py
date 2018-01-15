import tensorflow as tf
import tflearn.layers.conv
from Encoder import Encoder
import code

class QuadroEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)
        self.channels = [512,256,128]

    def encodeLevel(self, features, initial_state, direction='rows', level=0):
        '''
        Encodes one level of the input features bidirectionally.
        input1: B x H x W x C
        input2: 2 x B x N x C'
        output1: B x H x W x 2 * C'
        output2: 2 x B x X x 2 * C'
        output3: 2 x B x 4 * C'
        '''
        myscope = self.model.encoder + '_' + direction + '_' + str(level)
        shape = tf.shape(features)
        batchsize = shape[0]
        height = shape[1]
        width = shape[2]
        channels = features.shape[3].value
        # switches rows and cols in case that the encoding is done columnwise
        if direction == 'cols':
            features = tf.transpose(features, [0,2,1,3])
            x = width
            y = height
        else:
            x = height
            y = width
        # "flattens" the tensor in order to put it in the RNN
        features = tf.reshape(features, [batchsize * x, y, channels])
        # build actual lstm states
        initial_state = tf.reshape(initial_state, [2, batchsize * y, self.channels[level]])
        initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state[0],initial_state[1])
        # create the encoding LSTM
        with tf.variable_scope(scope=myscope + '_1', reuse=None):            
            rnncell_fw = tf.contrib.rnn.BasicLSTMCell(self.channels[level])
            rnncell_bw = tf.contrib.rnn.BasicLSTMCell(self.channels[level])
            # B * X x Y x C, 2 x B * X x C' -> 2 x B * X x Y x C', 2 x 2 x B * X x C'
            features, states = tf.nn.bidirectional_dynamic_rnn( \
                rnncell_fw, \
                rnncell_bw, \
                features, \
                initial_state_fw=initial_state, \
                initial_state_bw=initial_state, \
                parallel_iterations=1)
        # reorder and reshape dimensions from 2 x B * X x Y x C' to wanted B x H x W x 2 * C'
        features = tf.transpose(features, [1,2,0,3])
        local_features = tf.reshape(features, [batchsize, x, y, 2 * self.channels[channel]])
        if direction == 'cols':
            local_features = tf.transpose(local_features, [0,2,1,3])
        # reorder and reshape dimensions from 2 x 2 x B * X x C' to wanted 2 x B x X x 2 * C'
        c = tf.concat([states[0][0],states[0][1]],-1)
        c = tf.expand_dims(c,0)
        h = tf.concat([states[1][0],states[1][1]],-1)
        h = tf.expand_dims(h,0)
        intermediate_features = tf.concat([c,h],0)
        #
        intermediate_features_t = tf.concat(\
            intermediate_features[0],
            intermediate_features[1],
            -1)
        # create the "inner" LSTM
        with tf.variable_scope(scope=myscope + '_2', reuse=None):
            rnncell = tf.contrib.rnn.BasicLSTMCell(2 * self.channels)
            # B x X x 2 * 2 * C', 2 x B x 2 * 2 * C' ->  B x X x 2 * 2 * C', 2 x B x 2 * 2 * C'
            _, states = tf.nn.dynamic_rnn( \
                rnncell, \
                intermediate_features_t, \
                dtype=tf.float32, \
                parallel_iterations=1)
        #
        global_features = tf.concatenate(states[0],states[1],-1)
        # return all 3 levels of features
        return local_features, intermediate_features, global_features

    def generatePositionalEmbeddings(self, n, batchsize):
        '''
        Generates the initial states for the row/col encoding by encoding the row/col number.
        input: 1
        returns: 2 x B x N x C
        '''
        positional_embedding = tf.range(n)
        positional_embedding = tf.expand_dims(positional_embedding, 0)
        positional_embedding = tf.tile(positional_embedding, [self.channels[0], 1])
        #
        positional_embedding = tf.transpose(positional_embedding) # N x C
        positional_embedding = tf.expand_dims(positional_embedding, 0)# 1 x W x C
        positional_embedding = tf.tile(positional_embedding, [batchsize, 1, 1]) # B x W x C
        #
        positional_embedding = tf.reshape( \
            positional_embedding, \
            [batchsize*n, self.channels[0]])
        positional_embedding = tf.cast(positional_embedding, tf.float32)
        positional_embedding = tf.expand_dims(positional_embedding, 0)
        return tf.concatenate([positional_embedding,positional_embedding],0)
        

    def updateInitialStates(self, intermediate_features, level, ref, s=2):
        '''
        Updates the initial
        input: 2 x B x N x C
        returns: 2 x B x s*N x C'
        '''
        assert level < len(self.channels) - 1
        shape = tf.shape(features)
        batchsize = shape[0]
        n = shape[1]
        # unpool
        initial_state = tf.expand_dims(intermediate_features, 1)
        initial_state = tf.tile(initial_state, [s,1])
        initial_state = tf.transpose(initial_state, [0,2,1,3,4])
        initial_state = tf.reshape( \
            initial_state, \
            [batchsize, s * n, 2, self.channels[level]])
        # fit number of channels
        w = tf.get_variable( \
            'weight', \
            [self.channels[level], self.channels[level + 1]], \
            tf.float32, \
            tf.random_normal_initializer())
        b = tf.get_variable( \
            'bias', \
            [2 * self.channels[level]], \
            tf.float32, \
            tf.random_normal_initializer())
        initial_state = tf.tensordot(initial_state,w,[[-1],[0]]) + b1
        initial_state = tf.nn.tanh(initial_state)
        # trim to exact size of feature level
        return initial_state[batchsize, :ref, 2, self.channels[level + 1]]

    def unpool(self, features, level, sh=2, sw=2):
        '''
        Unpools the encoder features in order to be able to concatenate levels.
        input: B x H x W x C
        returns: B x sh*H x sw*W x C
        '''
        assert level < len(self.channels)
        shape = tf.shape(features)
        batchsize = shape[0]
        height = shape[1]
        width = shape[2]
        channels = features.shape[3]
        # vertical unpooling
        nfeatures = tf.expand_dims(features, 1)
        nfeatures = tf.tile(nfeatures, [sh,1])
        nfeatures = tf.transpose(nfeatures, [0,2,1,3,4])
        nfeatures = tf.reshape( \
            nfeatures, \
            [batchsize, sh * height, width, channels])
        # horizontal unpooling
        nfeatures = tf.expand_dims(nfeatures, 2)
        nfeatures = tf.tile(nfeatures, [sw,1])
        nfeatures = tf.transpose(nfeatures, [0,1,3,2,4])
        nfeatures = tf.reshape( \
            nfeatures, \
            [batchsize, sh * height, sw * width, channels])
        # trim to exact size of the feature level
        shape_ref = tf.shape(self.model.featureLevels[level])
        height_ref = shape_ref[1]
        width_ref = shape_ref[2]
        return nfeatures[batchsize, :height_ref, :width_ref, channels]


    def createGraph(self):
        '''
        '''
        assert len(self.channels) == len(self.model.featureLevels)
        shape = tf.shape(self.model.features)
        batchsize = shape[0]
        height = shape[1]
        width = shape[2]
        #
        features = self.model.featureLevels[0]
        initial_state_rows = self.generatePositionalEmbeddings(height, batchsize)
        initial_state_cols = self.generatePositionalEmbeddings(width, batchsize)
        #
        local_features = []
        global_features = []
        #
        for level in range(len(self.channels)):
            # rowwise encoding
            local_row_features, intermediate_row_features, global_row_features = \
                self.encodeLevel( \
                    self.model.featureLevels[level], \
                    initial_state_rows, \
                    direction='rows', \
                    level=level)
            # columnwise encoding
            local_col_features, intermediate_col_features, global_col_features = \
                self.encodeLevel(
                    features_in, \
                    initial_state_cols, \
                    direction='cols', \
                    level=level)
            #
            if level == 0:
                local_features = tf.concat( \
                    [local_row_features, local_col_features], -1)
                global_features = tf.concat( \
                    [global_row_features, global_row_features], -1)
            else:
                local_features = tf.concat( \
                    [local_features, local_row_features, local_col_features], -1)
                global_features = tf.concat( \
                    [global_features, global_row_features, global_row_features], -1)
            #
            if level < len(self.model.featureLevels) - 1:
                local_features = self.unpool(local_features, level)
                features_in = tf.concat( \
                    [local_features, self.model.feature_levels[level + 1]], -1)
                initial_state_rows = self.updateInitialStates( \
                    intermediate_row_features, \
                    level, \
                    tf.shape(self.model.featureLevels[level + 1])[1])
                initial_state_cols = self.updateInitialStates( \
                    intermediate_col_features, \
                    level, \
                    tf.shape(self.model.featureLevels[level + 1])[2])
        #
        self.global_features = global_features
        self.local_features = local_features