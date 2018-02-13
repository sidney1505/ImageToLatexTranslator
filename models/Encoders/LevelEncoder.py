import tensorflow as tf
import tflearn.layers.conv, math
from Encoder import Encoder
import code
# code.interact(local=dict(globals(), **locals()))

class LevelEncoder(Encoder):
    def __init__(self, model):
        Encoder.__init__(self, model)
        encoder_size = self.model.encoder_size # prev 64
        # version a
        self.channels =  [4 * encoder_size, 2 * encoder_size, 2 * encoder_size]
        # version b
        # self.channels =  [4 * encoder_size, 2 * encoder_size, encoder_size]
        # version a2
        # self.channels =  [2 * encoder_size, encoder_size]

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
        channelsx = self.channels[level]
        #
        initial_state = tf.reshape(initial_state, [2, batchsize * x, channelsx])
        initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state[0],initial_state[1])
        # create the encoding LSTM
        with tf.variable_scope(myscope + '_1', reuse=None):
            rnncell_fw = tf.contrib.rnn.BasicLSTMCell(channelsx)
            rnncell_bw = tf.contrib.rnn.BasicLSTMCell(channelsx)
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
        local_features = tf.reshape(features, [batchsize, x, y, 2 * channelsx])
        if direction == 'cols':
            local_features = tf.transpose(local_features, [0,2,1,3])
        # reorder and reshape dimensions from 2 x 2 x B * X x C' to wanted 2 x B x X x 2 * C'
        c = tf.concat([states[0][0],states[0][1]],-1)
        c = tf.expand_dims(c,0)
        h = tf.concat([states[1][0],states[1][1]],-1)
        h = tf.expand_dims(h,0)
        intermediate_features = tf.concat([c,h],0)
        intermediate_features = tf.reshape(
            intermediate_features, [2, batchsize, x, 2 * channelsx])
        #
        intermediate_features_t = tf.concat( \
            [intermediate_features[0], intermediate_features[1]], -1)
        # create the "inner" LSTM
        with tf.variable_scope(myscope + '_2', reuse=None):
            rnncell = tf.contrib.rnn.BasicLSTMCell(2 * channelsx)
            # B x X x 2 * 2 * C', 2 x B x 2 * 2 * C' ->  B x X x 2 * 2 * C', 2 x B x 2 * 2 * C'
            _, states = tf.nn.dynamic_rnn( \
                rnncell, \
                intermediate_features_t, \
                dtype=tf.float32, \
                parallel_iterations=1)
        #
        c = tf.expand_dims(states[0],0)
        h = tf.expand_dims(states[1],0)
        global_features = tf.concat([c,h],0)
        # return all 3 levels of features
        return local_features, intermediate_features, global_features

    def generatePositionalEmbeddings(self, n, batchsize, level=0):
        '''
        Generates the initial states for the row/col encoding by encoding the row/col number.
        input: 1
        returns: 2 x B x N x C
        '''
        positional_embedding = tf.range(n)
        positional_embedding = tf.expand_dims(positional_embedding, 0)
        positional_embedding = tf.tile(positional_embedding, [self.channels[level], 1])
        #
        positional_embedding = tf.transpose(positional_embedding) # N x C
        positional_embedding = tf.expand_dims(positional_embedding, 0)# 1 x W x C
        positional_embedding = tf.tile(positional_embedding, [batchsize, 1, 1]) # B x W x C
        #
        positional_embedding = tf.reshape( \
            positional_embedding, \
            [batchsize*n, self.channels[level]])
        positional_embedding = tf.cast(positional_embedding, tf.float32)
        positional_embedding = tf.expand_dims(positional_embedding, 0)
        return tf.concat([positional_embedding,positional_embedding],0)
        

    def updateInitialStates(self, intermediate_features, level, ref, direction='row', s=2):
        '''
        Updates the initial
        input: 2 x B x X x C
        returns: 2 x B x s * X x C'
        '''
        assert level < len(self.channels) - 1
        shape = tf.shape(intermediate_features)
        batchsize = shape[1]
        n = shape[2]
        current_channels = intermediate_features.shape[3].value
        # unpool
        initial_state = tf.expand_dims(intermediate_features, 2)
        initial_state = tf.tile(initial_state, [1,1,s,1,1])
        initial_state = tf.transpose(initial_state, [0,1,3,2,4])
        initial_state = tf.reshape( \
            initial_state, \
            [2, batchsize, s * n, current_channels])
        # fit number of channels
        # the '- 1' is because the positional embedding is concatenated later
        myscope = 'updateInitialStates_' + direction + '_' + str(level)
        with tf.variable_scope(myscope, reuse=None):
            w = tf.get_variable( \
                'weight', \
                [current_channels, self.channels[level + 1] - 1], \
                tf.float32, \
                tf.random_normal_initializer())
            b = tf.get_variable( \
                'bias', \
                [self.channels[level + 1] - 1], \
                tf.float32, \
                tf.random_normal_initializer())
            initial_state = tf.tensordot(initial_state,w,[[-1],[0]]) + b
        initial_state = tf.nn.tanh(initial_state)
        # trim to exact size of feature level
        initial_state = initial_state[:2, :batchsize, :ref]
        # append new positional embedding
        new_n = tf.shape(initial_state)[2]
        positional_embedding = tf.range(new_n)
        positional_embedding = tf.expand_dims(positional_embedding, 1)
        positional_embedding = tf.expand_dims(positional_embedding, 0)
        positional_embedding = tf.expand_dims(positional_embedding, 0)
        positional_embedding = tf.tile(positional_embedding, [2, batchsize, 1, 1]) # ???
        positional_embedding = tf.cast(positional_embedding, tf.float32)
        initial_state = tf.concat([initial_state,positional_embedding],-1)
        return initial_state

    def unpool(self, features, level, sh=2, sw=2):
        '''
        Unpools the encoder features in order to be able to concatenate levels.
        input: B x H x W x C
        returns: B x sh*H x sw*W x C
        '''
        assert level < len(self.channels) - 1
        shape = tf.shape(features)
        batchsize = shape[0]
        height = shape[1]
        width = shape[2]
        channels = features.shape[3].value
        # vertical unpooling
        nfeatures = tf.expand_dims(features, 1)
        nfeatures = tf.tile(nfeatures, [1,sh,1,1,1])
        nfeatures = tf.transpose(nfeatures, [0,2,1,3,4])
        nfeatures = tf.reshape( \
            nfeatures, \
            [batchsize, sh * height, width, channels])
        # horizontal unpooling
        nfeatures = tf.expand_dims(nfeatures, 2)
        nfeatures = tf.tile(nfeatures, [1,1,sw,1,1])
        nfeatures = tf.transpose(nfeatures, [0,1,3,2,4])
        nfeatures = tf.reshape( \
            nfeatures, \
            [batchsize, sh * height, sw * width, channels])
        # trim to exact size of the feature level
        shape_ref = tf.shape(self.model.featureLevels[level + 1])
        height_ref = shape_ref[1]
        width_ref = shape_ref[2]
        return nfeatures[:batchsize, :height_ref, :width_ref]


    def createGraph(self):
        '''
        '''
        #assert len(self.channels) == len(self.model.featureLevels)
        shape = tf.shape(self.model.featureLevels[0])
        batchsize = shape[0]
        height = shape[1]
        width = shape[2]
        #
        is_version_a = True
        #
        features_in = self.model.featureLevels[0]
        local_features = None
        global_features = None
        #
        for level in range(len(self.channels)):
            # version b
            nheight = tf.shape(features_in)[1]
            nwidth = tf.shape(features_in)[2]
            if level == 0 or not is_version_a:
                initial_state_rows = \
                    self.generatePositionalEmbeddings(nheight, batchsize, level)
                initial_state_cols = \
                    self.generatePositionalEmbeddings(nwidth, batchsize, level)
            # rowwise encoding
            local_row_features, intermediate_row_features, global_row_features = \
                self.encodeLevel( \
                    features_in, \
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
                    [global_features, global_row_features, global_col_features], -1)
            #
            if level < len(self.model.featureLevels) - 1:
                local_features = self.unpool(local_features, level)
                features_in = tf.concat( \
                    [local_features, self.model.featureLevels[level + 1]], -1)
                if is_version_a:
                    initial_state_rows = self.updateInitialStates( \
                        intermediate_row_features, \
                        level, \
                        tf.shape(self.model.featureLevels[level + 1])[1], \
                        'rows')
                    initial_state_cols = self.updateInitialStates( \
                        intermediate_col_features, \
                        level, \
                        tf.shape(self.model.featureLevels[level + 1])[2], \
                        'cols')
        #
        shape_fine = tf.shape(self.model.featureLevels[-1])
        self.model.encoded_batchsize = shape_fine[0]
        self.model.encoded_height = shape_fine[1]
        self.model.encoded_width = shape_fine[2]
        self.model.encoded_channels = local_features.shape[-1].value
        self.model.refined_features = tf.reshape( \
            local_features, \
            [ \
                self.model.encoded_batchsize, \
                self.model.encoded_height * self.model.encoded_width, \
                self.model.encoded_channels
            ])
        self.model.input_summary = tf.contrib.rnn.LSTMStateTuple( \
            global_features[0], global_features[1])