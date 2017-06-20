import code
import os
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

class Model:
    def __init__(self,  num_classes, max_num_tokens, minibatchsize=5, learning_rate=0.001, num_features=512, nr_epochs=50, model_dir=''):
        # intialise class variables
        self.num_classes = num_classes
        self.max_num_tokens = max_num_tokens
        self.minibatchsize = minibatchsize
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.nr_epochs = nr_epochs
        # create the network graph
        self.images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        network = self.createCNNModel(self.images_placeholder)
        network, self.num_features = self.createEncoderLSTM(network)
        network = self.createDecoderLSTM(network)
        self.network = network
        # save the model
        self.model_dir = model_dir
        self.train_logs = self.model_dir + '/train_logs'
        if not os.path.exists(self.train_logs):
            os.makedirs(self.train_logs)
        self.validation_logs = self.model_dir + '/validation_logs'
        if not os.path.exists(self.validation_logs):
            os.makedirs(self.validation_logs)
        self.save_path = self.model_dir + '/weights.cpk'
        self.param_path = self.model_dir + '/params.npz'
        if not os.path.exists(self.param_path):
            with open(self.param_path, 'w') as fout:
                np.savez(fout, num_classes=num_classes, max_num_tokens=max_num_tokens, minibatchsize=minibatchsize, learning_rate=learning_rate, num_features=num_features, nr_epochs=nr_epochs, model_dir=model_dir)
        self.saver = tf.train.Saver()


    def createCNNModel(self, network):
        network = tflearn.layers.conv.conv_2d(network, 64, 3, activation='relu') # padding???
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 128, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 256, 3, activation='relu')
        network = tflearn.layers.normalization.batch_normalization(network) #same as torch?

        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, [1,2], strides=2)

        network = conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.normalization.batch_normalization(network) #same as torch?
        network = max_pool_2d(network, [2,1], strides=2)

        network = conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.normalization.batch_normalization(network) #same as torch?
        #network = tf.Print(network,[tf.shape(network)],"after cnn: ")
        return network

    def createEncoderLSTM(self, network):
        batchsize = tf.shape(network)[0]
        network = tf.transpose(network, [1,0,2,3])
        rnncell_fw = tf.contrib.rnn.LSTMCell(256) # TODO choose parameter
        # trainable hidden state??? (postional embedding)
        fw_state = rnncell_fw.zero_state(batch_size=batchsize, dtype=tf.float32)
        rnncell_bw = tf.contrib.rnn.LSTMCell(256) # TODO choose parameter
        bw_state = rnncell_bw.zero_state(batch_size=batchsize, dtype=tf.float32)
        l = tf.TensorArray(dtype=tf.float32, size=tf.shape(network)[0])
        params = [tf.constant(0), network, l, fw_state, bw_state]
        rows = tf.shape(network)[0]
        #rows = tf.Print(rows,[rows],"row count: ")
        while_condition = lambda i, network, l, fw_state, bw_state: tf.less(i, rows)
        def body(i, network, l, fw_state, bw_state):
            #i = tf.Print(i,[i],"row: ")
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(rnncell_fw, rnncell_bw, network[i], initial_state_fw=fw_state,initial_state_bw=bw_state) #,dtype=tf.float32)
            fw_state, bw_state = output_states
            # code.interact(local=locals())
            l = l.write(i, outputs)        
            return [tf.add(i, 1), network, l, fw_state, bw_state]
        i, network, l, fw_state, bw_state = tf.while_loop(while_condition, body, params)
        network = l.stack()
        #network = tf.Print(network,[tf.shape(network)],"in encoder: ")
        network = tf.transpose(network, perm=[2,0,3,1,4])
        s = tf.shape(network)
        # code.interact(local=locals())
        num_features = (network.shape[3] * network.shape[4]).value # other solution???
        network = tf.reshape(network, [s[0],s[1],s[2],s[3]*s[4]])
        #network = tf.Print(network,[tf.shape(network)],"after encoder: ")
        return network, num_features #

    def createDecoderLSTM(self, network):
        shape = tf.Print(tf.shape(network),[tf.shape(network)],"dynamic network shape from decoder input!!!!!!",1)
        dim_beta = 50 # hyperparameter!!!
        dim_h = 512 # hyperparameter!!!
        batchsize = shape[0] # tf.shape(network)[0] # besserer weg???
        # num_features = network.shape[3]
        dim_o = self.num_features + dim_h
        # the used rnncell
        rnncell = tf.contrib.rnn.LSTMCell(256, state_is_tuple=False, reuse=None) #size is hyperparamter!!!
        # used weight variables
        beta = tf.Variable(tf.random_normal([dim_beta])) # size is hyperparamter!!!
        w1 = tf.Variable(tf.random_normal([dim_beta, dim_h])) # dim_beta x dim_h
        w2 = tf.Variable(tf.random_normal([self.num_features, dim_beta])) # num_features x dim_beta
        wc = tf.Variable(tf.random_normal([dim_o])) # (num_features + dim_h) (x hyperparam) ??? (= dim_o) ???
        wout = tf.Variable(tf.random_normal([self.num_classes, dim_o])) # num_classes x dim_o
        # initial states, ebenfalls Variablen???
        h0 = tf.zeros([batchsize,dim_h]) # 
        y0 = tf.zeros([batchsize,self.num_classes]) #
        o0 = tf.zeros([batchsize,dim_o]) #
        # brings network in shape for element wise multiplikation
        network_ = tf.expand_dims(network,4)
        # create necessaties for the while-loop
        l = tf.TensorArray(dtype=tf.float32, size=self.max_num_tokens)
        params = [tf.constant(0), network, l, h0, y0, o0, beta, network_, w1, w2]
        tmax = tf.constant(self.max_num_tokens)
        #tmax = tf.Print(tmax,[tmax],"tmax =============================")
        def while_condition(t, network, l, h, y, o, beta, network_, w1, w2):
            #t = tf.Print(t,[t],"twhile =============================")
            return tf.less(t, tmax) # token embedding??
        def body(t, network, l, h, y, o, beta, network_, w1, w2):
#            cond = lambda t: tf.equal(t,tf.constant(9))
#            def aaa(t, network, l, h, y, o, beta, network_, w1, w2):
#                t = tf.add(t, 1)
#                t = tf.Print(t,[t],"tnew =============================")
#                return [t, network, l, h, y, o, beta, network_, w1, w2]
#            #def bbb(t, network, l, h, y, o, beta, network_, w1, w2):
            #t = tf.Print(t,[t],"t0 =============================")
            hdotw1 = tf.tensordot(h,w1,[[1],[1]])
            hdotw1 = tf.expand_dims(hdotw1,1)
            hdotw1 = tf.expand_dims(hdotw1,1)
            hdotw1 = tf.expand_dims(hdotw1,1)
            e = tf.tensordot(beta,tf.tanh(hdotw1 + w2 * network_),[[0],[4]]) # batchsize x height x width x num_features # shapes angleichen?? batches??
            #e = tf.Print(e,[tf.shape(e)],"e =============================")
            #t = tf.Print(t,[t],"t1 =============================")
            # code.interact(local=locals())
            alpha = tf.nn.softmax(e) # batchsize x height x width x num_features # zeilenweise normieren!!!
            #alpha = tf.Print(alpha,[tf.shape(alpha)],"t =============================")
            #t = tf.Print(t,[t],"t2 =============================")
            c = alpha * network
            c = tf.transpose(c, perm=[1,2,0,3]) # put rows and columns in front in order to sum over them
            # sums over rows
            c = tf.foldl(lambda a, x: a + x, c)
            # sums over columns
            c = tf.foldl(lambda a, x: a + x, c) # batchsize x num_features
            #c = tf.Print(c,[tf.shape(c)],"c =============================")
            #t = tf.Print(t,[t],"t3 =============================")
            oput,h = rnncell.__call__(tf.concat([y,o],1), h) # batchsize x dim_h  # cell state & input als parameter uebergeben!!!
            #h = tf.Print(h,[tf.shape(h)],"h =============================")
            #t = tf.Print(t,[t],"t4 =============================")
            # code.interact(local=locals())
            o = tf.tanh(wc * tf.concat([h,c],1)) # batchsize x dim_o
            #o = tf.Print(o,[tf.shape(o)],"o =============================")
            #t = tf.Print(t,[t],"t5 =============================")
            y = tf.nn.softmax(tf.tensordot(o,wout,[[1],[1]])) # batch_size x num_classes # zeilenweise normieren!!!
            #y = tf.Print(y,[tf.shape(y)],"y =============================")
            #t = tf.Print(t,[t],"t6 =============================")
            l = l.write(t, y)
            #t = tf.Print(t,[t],"t7 =============================")
            t = tf.add(t, 1)
            #t = tf.Print(t,[t],"tnew =============================")
            return [t, network, l, h, y, o, beta, network_, w1, w2]
            #return tf.cond(cond,aaa,bbb)
        t, network, l, h, y, o, beta, network_, w1, w2 = tf.while_loop(while_condition, body, params)
        #t = tf.Print(t,[t],"ttttttttttttttttttttttttttttend =============================",5)
        l = l.stack()
        l = tf.transpose(l, [1,0,2])
        #l = tf.Print(l,[tf.shape(l)],"after decoder: ")
        return l

    def loss(self, pred, labels):
        pred = tf.transpose(pred, [1,0,2])
        labels = tf.transpose(labels, [1,0])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=pred)
        return tf.reduce_mean(loss)

    def training(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss) #, global_step=global_step)
        return train_op

    def addTrainLog(self, loss_value, epoch, batch):
        with open(self.train_logs+'/log_'+str(epoch)+'_'+str(batch)+'.npz', 'w') as fout:
            np.savez(fout,val=np.mean(loss_value))

    def addValidationLog(self, loss_value, epoch, batch):
        with open(self.validation_logs+'/log_'+str(epoch)+'_'+str(batch)+'.npz', 'w') as fout:
            np.savez(fout,val=np.mean(loss_value))

    def save(self, sess):
        self.saver.save(sess, self.save_path)

    def load(model_path):