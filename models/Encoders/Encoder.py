import tensorflow as tf
import code

class Encoder:
    def __init__(self, model):
        self.model = model

    def createGraph(self):
        raise NotImplementedError

    def encodeRows(self):
    	with tf.variable_scope(self.model.encoder + '_rows', reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            height = shape[1]
            width = shape[2]
            num_features = self.model.features.shape[3].value
            # rows
            rowrnncell = tf.contrib.rnn.BasicLSTMCell(self.channels)
            rowfeatures = tf.reshape(self.model.features, [batchsize * height, width, \
                num_features])
            # create the postional row embeddings
            
            initial_state_rows = tf.range(height)
            initial_state_rows = tf.expand_dims(initial_state_rows, 0)
            initial_state_rows = tf.tile(initial_state_rows, [self.channels, 1])

            initial_state_rows = tf.transpose(initial_state_rows) # WxC
            initial_state_rows = tf.expand_dims(initial_state_rows, 0)# 1xWxC
            initial_state_rows = tf.tile(initial_state_rows, [batchsize, 1, 1]) # BxWxC
                                    
            initial_state_rows = tf.reshape(initial_state_rows, [batchsize*height, \
            	self.channels])
            initial_state_rows = tf.cast(initial_state_rows, tf.float32)
            initial_state_rows = tf.contrib.rnn.LSTMStateTuple(initial_state_rows, \
                initial_state_rows)
            # run the actual row encoding lstm
            #code.interact(local=dict(globals(), **locals()))
            rowfeatures, rowstates = tf.nn.dynamic_rnn(rowrnncell, rowfeatures, \
                initial_state=initial_state_rows, parallel_iterations=1)

        with tf.variable_scope(self.model.encoder + '_rows2', reuse=None):
            #
            self.refined_rows = tf.reshape(rowfeatures, [batchsize,height,width,\
                self.channels]) # (20, 4, 48, 2048)            
            #
            rowstates_c = tf.reshape(rowstates[0], [batchsize, height, self.channels])
            rowstates_h = tf.reshape(rowstates[1], [batchsize, height, self.channels])
            rowstates = tf.concat([rowstates_c, rowstates_h],-1)
            #
            w1 = tf.get_variable('weight', [2 * self.channels, self.channels], tf.float32, \
                    tf.random_normal_initializer())
            b1 = tf.get_variable('bias', [self.channels], tf.float32, \
                    tf.random_normal_initializer())
            rowstates = tf.tensordot(rowstates,w1,[[-1],[0]]) + b1
            rowstates = tf.nn.tanh(rowstates)
            rowstates = tf.nn.dropout(rowstates,self.model.keep_prob)
            # 
            rowstaternncell = tf.contrib.rnn.BasicLSTMCell(self.channels)
            _, rowstates = tf.nn.dynamic_rnn(rowstaternncell, rowstates, \
                dtype=tf.float32, parallel_iterations=1)
            rowstates = tf.reshape(rowstates, [2, batchsize, self.channels])
            self.row_summary = tf.contrib.rnn.LSTMStateTuple(rowstates[0], \
                rowstates[1])

    def encodeRowsBidirectional(self):
        with tf.variable_scope(self.model.encoder + '_rows', reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            height = shape[1]
            width = shape[2]
            num_features = self.model.features.shape[3].value
            # rows            
            rowfeatures = tf.reshape(self.model.features, [batchsize * height, width, \
                num_features])
            # create the postional row embeddings
            
            initial_state_rows = tf.range(height)
            initial_state_rows = tf.expand_dims(initial_state_rows, 0)
            initial_state_rows = tf.tile(initial_state_rows, [self.channels, 1])

            initial_state_rows = tf.transpose(initial_state_rows) # WxC
            initial_state_rows = tf.expand_dims(initial_state_rows, 0)# 1xWxC
            initial_state_rows = tf.tile(initial_state_rows, [batchsize, 1, 1]) # BxWxC
                                    
            initial_state_rows = tf.reshape(initial_state_rows, [batchsize*height, \
                self.channels])
            initial_state_rows = tf.cast(initial_state_rows, tf.float32)
            initial_state_rows = tf.contrib.rnn.LSTMStateTuple(initial_state_rows, \
                initial_state_rows)
            # run the actual row encoding lstm
            #code.interact(local=dict(globals(), **locals()))
            rowrnncell_fw = tf.contrib.rnn.BasicLSTMCell(self.channels)
            rowrnncell_bw = tf.contrib.rnn.BasicLSTMCell(self.channels)
            rowfeatures, rowstates = tf.nn.bidirectional_dynamic_rnn(rowrnncell_fw, \
                rowrnncell_bw, rowfeatures, initial_state_fw=initial_state_rows, \
                initial_state_bw=initial_state_rows, parallel_iterations=1)
            rowfeatures = tf.concat([rowfeatures[0],rowfeatures[1]],-1)
            c = tf.concat([rowstates[0][0],rowstates[0][1]],-1)
            h = tf.concat([rowstates[1][0],rowstates[1][1]],-1)
            rowstates = tf.contrib.rnn.LSTMStateTuple(c, h)
            # code.interact(local=dict(globals(), **locals()))

        with tf.variable_scope(self.model.encoder + '_rows2', reuse=None):
            #
            self.refined_rows = tf.reshape(rowfeatures, [batchsize,height,width,\
                2 * self.channels]) # (20, 4, 48, 2048)            
            #
            rowstates_c = tf.reshape(rowstates[0], [batchsize, height, 2 * self.channels])
            rowstates_h = tf.reshape(rowstates[1], [batchsize, height, 2 * self.channels])
            rowstates = tf.concat([rowstates_c, rowstates_h],-1)
            #
            w1 = tf.get_variable('weight', [4 * self.channels, 2 * self.channels], tf.float32, \
                    tf.random_normal_initializer())
            b1 = tf.get_variable('bias', [2 * self.channels], tf.float32, \
                    tf.random_normal_initializer())
            rowstates = tf.tensordot(rowstates,w1,[[-1],[0]]) + b1
            rowstates = tf.nn.tanh(rowstates)
            rowstates = tf.nn.dropout(rowstates,self.model.keep_prob)
            # 
            rowstaternncell = tf.contrib.rnn.BasicLSTMCell(2 * self.channels)
            _, rowstates = tf.nn.dynamic_rnn(rowstaternncell, rowstates, \
                dtype=tf.float32, parallel_iterations=1)
            rowstates = tf.reshape(rowstates, [2, batchsize, 2 * self.channels])
            self.row_summary = tf.contrib.rnn.LSTMStateTuple(rowstates[0], rowstates[1])

    def encodeRowsBidirectionalStacked(self, number_of_layers):
        with tf.variable_scope(self.model.encoder + '_rows', reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            height = shape[1]
            width = shape[2]
            num_features = self.model.features.shape[3].value
            # rows            
            rowfeatures = tf.reshape(self.model.features, [batchsize * height, width, \
                num_features])
            # create the postional row embeddings
            
            initial_state_rows = tf.range(height)
            initial_state_rows = tf.expand_dims(initial_state_rows, 0)
            initial_state_rows = tf.tile(initial_state_rows, [self.channels, 1])

            initial_state_rows = tf.transpose(initial_state_rows) # WxC
            initial_state_rows = tf.expand_dims(initial_state_rows, 0)# 1xWxC
            initial_state_rows = tf.tile(initial_state_rows, [batchsize, 1, 1]) # BxWxC
                                    
            initial_state_rows = tf.reshape(initial_state_rows, [batchsize*height, \
                self.channels])
            initial_state_rows = tf.cast(initial_state_rows, tf.float32)
            initial_state_rows = tf.contrib.rnn.LSTMStateTuple(initial_state_rows, \
                initial_state_rows)
            # run the actual row encoding lstm
            #code.interact(local=dict(globals(), **locals()))
            def createMulticell():
                cells = []
                initial_states = []
                for i in range(number_of_layers):
                    cells.append(tf.contrib.rnn.BasicLSTMCell(self.channels))
                    initial_states.append(initial_state_rows)
                initial_states = tuple(initial_states)
                multicell = tf.contrib.rnn.MultiRNNCell(cells)
                return multicell, initial_states
            rowrnncell_fw, initial_state_rows = createMulticell()
            rowrnncell_bw, _ = createMulticell()
            rowfeatures, rowstates = tf.nn.bidirectional_dynamic_rnn(rowrnncell_fw, \
                rowrnncell_bw, rowfeatures, initial_state_fw=initial_state_rows, \
                initial_state_bw=initial_state_rows, parallel_iterations=1)
            rowfeatures = tf.concat([rowfeatures[0],rowfeatures[1]],-1)
            nrowstates = []
            w1 = tf.get_variable('weight', [4 * self.channels, 2 * self.channels], \
                    tf.float32, tf.random_normal_initializer())
            b1 = tf.get_variable('bias', [2 * self.channels], tf.float32, \
                    tf.random_normal_initializer())
            for i in range(number_of_layers):
                s = tf.concat([rowstates[i][0][0],rowstates[i][0][1],rowstates[i][1][0],\
                    rowstates[i][1][1]],-1)
                s = tf.tensordot(s,w1,[[-1],[0]]) + b1
                s = tf.nn.tanh(s)
                s = tf.nn.dropout(s,self.model.keep_prob)
                nrowstates.append(s)
            rowstates = nrowstates
            # code.interact(local=dict(globals(), **locals()))

        with tf.variable_scope(self.model.encoder + '_rows2', reuse=None):
            # 
            rowstaternncell = tf.contrib.rnn.BasicLSTMCell(2 * self.channels)
            # use trick with batches
            _, rowstates = tf.nn.dynamic_rnn(rowstaternncell, rowstates, \
                dtype=tf.float32, parallel_iterations=1)
            row_summary = []
            for i in range(number_of_layers):
                r = tf.reshape(rowstates, [2, batchsize, 2 * self.channels])
                row_summary.append(tf.contrib.rnn.LSTMStateTuple(r[0], r[1]))
            self.row_summary = row_summary


    def encodeCols(self):
        with tf.variable_scope(self.model.encoder + '_cols', reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            height = shape[1]
            width = shape[2]
            num_features = self.model.features.shape[3].value
            # cols
            colrnncell = tf.contrib.rnn.BasicLSTMCell(self.channels)
            colfeatures = tf.transpose(self.model.features, [0,2,1,3])
            colfeatures = tf.reshape(self.model.features, [batchsize * width, height, \
                num_features])
            # create the postional col embeddings
            
            initial_state_cols = tf.range(width)
            initial_state_cols = tf.expand_dims(initial_state_cols, 0)
            initial_state_cols = tf.tile(initial_state_cols, [self.channels, 1])

            initial_state_cols = tf.transpose(initial_state_cols) # WxC
            initial_state_cols = tf.expand_dims(initial_state_cols, 0)# 1xWxC
            initial_state_cols = tf.tile(initial_state_cols, [batchsize, 1, 1]) # BxWxC
                                    
            initial_state_cols = tf.reshape(initial_state_cols, [batchsize*width, \
            	self.channels])
            initial_state_cols = tf.cast(initial_state_cols, tf.float32)
            initial_state_cols = tf.contrib.rnn.LSTMStateTuple(initial_state_cols, \
                initial_state_cols)
            # run the actual col encoding lstm
            #code.interact(local=dict(globals(), **locals()))
            colfeatures, colstates = tf.nn.dynamic_rnn(colrnncell, colfeatures, \
                initial_state=initial_state_cols, parallel_iterations=1)

        with tf.variable_scope(self.model.encoder + '_cols2', reuse=None):
            #
            colfeatures = tf.reshape(colfeatures, [batchsize,width,height,\
                self.channels]) # (20, 4, 48, 2048)
            self.refined_cols = tf.transpose(colfeatures, [0,2,1,3])
            #
            colstates_c = tf.reshape(colstates[0], [batchsize, width, self.channels])
            colstates_h = tf.reshape(colstates[1], [batchsize, width, self.channels])
            colstates = tf.concat([colstates_c, colstates_h],-1)
            #
            w1 = tf.get_variable('weight', [2 * self.channels, self.channels], tf.float32, \
                    tf.random_normal_initializer())
            b1 = tf.get_variable('bias', [self.channels], tf.float32, \
                    tf.random_normal_initializer())
            colstates = tf.tensordot(colstates,w1,[[-1],[0]]) + b1
            colstates = tf.nn.tanh(colstates)
            colstates = tf.nn.dropout(colstates,self.model.keep_prob)
            # 
            colstaternncell = tf.contrib.rnn.BasicLSTMCell(self.channels)
            _, colstates = tf.nn.dynamic_rnn(colstaternncell, colstates, \
                dtype=tf.float32, parallel_iterations=1)
            colstates = tf.reshape(colstates, [2, batchsize, self.channels])
            self.col_summary = tf.contrib.rnn.LSTMStateTuple(colstates[0], \
                colstates[1])

    def encodeColsBidirectional(self):
        with tf.variable_scope(self.model.encoder + '_cols', reuse=None):
            shape = tf.shape(self.model.features)
            batchsize = shape[0]
            height = shape[1]
            width = shape[2]
            num_features = self.model.features.shape[3].value
            # cols
            
            colfeatures = tf.transpose(self.model.features, [0,2,1,3])
            colfeatures = tf.reshape(self.model.features, [batchsize * width, height, \
                num_features])
            # create the postional col embeddings
            
            initial_state_cols = tf.range(width)
            initial_state_cols = tf.expand_dims(initial_state_cols, 0)
            initial_state_cols = tf.tile(initial_state_cols, [self.channels, 1])

            initial_state_cols = tf.transpose(initial_state_cols) # WxC
            initial_state_cols = tf.expand_dims(initial_state_cols, 0)# 1xWxC
            initial_state_cols = tf.tile(initial_state_cols, [batchsize, 1, 1]) # BxWxC
                                    
            initial_state_cols = tf.reshape(initial_state_cols, [batchsize*width, \
                self.channels])
            initial_state_cols = tf.cast(initial_state_cols, tf.float32)
            initial_state_cols = tf.contrib.rnn.LSTMStateTuple(initial_state_cols, \
                initial_state_cols)
            # run the actual col encoding lstm
            #code.interact(local=dict(globals(), **locals()))
            colrnncell_fw = tf.contrib.rnn.BasicLSTMCell(self.channels)
            colrnncell_bw = tf.contrib.rnn.BasicLSTMCell(self.channels)
            colfeatures, colstates = tf.nn.bidirectional_dynamic_rnn(colrnncell_fw, \
                colrnncell_bw, colfeatures, initial_state_fw=initial_state_cols, \
                initial_state_bw=initial_state_cols, parallel_iterations=1)
            colfeatures = tf.concat([colfeatures[0],colfeatures[1]],-1)
            c = tf.concat([colstates[0][0],colstates[0][1]],-1)
            h = tf.concat([colstates[1][0],colstates[1][1]],-1)
            colstates = tf.contrib.rnn.LSTMStateTuple(c, h)

        with tf.variable_scope(self.model.encoder + '_cols2', reuse=None):
            #
            colfeatures = tf.reshape(colfeatures, [batchsize,width,height,\
                2 * self.channels]) # (20, 4, 48, 2048)
            self.refined_cols = tf.transpose(colfeatures, [0,2,1,3])
            #
            colstates_c = tf.reshape(colstates[0], [batchsize, width, 2 * self.channels])
            colstates_h = tf.reshape(colstates[1], [batchsize, width, 2 * self.channels])
            colstates = tf.concat([colstates_c, colstates_h],-1)
            #
            w1 = tf.get_variable('weight', [4 * self.channels, 2 * self.channels], tf.float32, \
                    tf.random_normal_initializer())
            b1 = tf.get_variable('bias', [2 * self.channels], tf.float32, \
                    tf.random_normal_initializer())
            colstates = tf.tensordot(colstates,w1,[[-1],[0]]) + b1
            colstates = tf.nn.tanh(colstates)
            colstates = tf.nn.dropout(colstates,self.model.keep_prob)
            # 
            colstaternncell = tf.contrib.rnn.BasicLSTMCell(2 * self.channels)
            _, colstates = tf.nn.dynamic_rnn(colstaternncell, colstates, \
                dtype=tf.float32, parallel_iterations=1)
            colstates = tf.reshape(colstates, [2, batchsize, 2 * self.channels])
            self.col_summary = tf.contrib.rnn.LSTMStateTuple(colstates[0], \
                colstates[1])