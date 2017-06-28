import code
import os
import shutil
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

class Model:
    def __init__(self,  num_classes, max_num_tokens, model_dir,\
            capacity=30000, learning_rate=0.01,\
            num_features=512, nr_epochs=50):
        # intialise class variables
        self.num_classes = num_classes
        self.max_num_tokens = max_num_tokens
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.nr_epochs = nr_epochs
        # save the model
        self.model_dir = model_dir
        self.logs = self.model_dir + '/logs'
        if not os.path.exists(self.logs):
            os.makedirs(self.logs)
        self.save_path = self.model_dir + '/weights.cpk'
        self.param_path = self.model_dir + '/params.npz'
        if not os.path.exists(self.param_path):
            with open(self.param_path, 'w') as fout:
                np.savez(fout, num_classes=num_classes, \
                    max_num_tokens=max_num_tokens, \
                    capacity=capacity, learning_rate=learning_rate, \
                    num_features=num_features, nr_epochs=nr_epochs, \
                    model_dir=model_dir)      
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wconv1': tf.Variable(tf.random_normal([3, 3, 1, 64]), name='wconv1'),
            'bconv1': tf.Variable(tf.random_normal([64]), name='bconv1'),
            # 5x5 conv, 32 inputs, 64 outputs
            'wconv2': tf.Variable(tf.random_normal([3, 3, 64, 128]), name='wconv2'),
            'bconv2': tf.Variable(tf.random_normal([128]), name='bconv2'),
            # 5x5 conv, 1 input, 32 outputs
            'wconv3': tf.Variable(tf.random_normal([3, 3, 128, 256]), name='wconv3'),
            'bconv3': tf.Variable(tf.random_normal([256]), name='bconv3'),
            # 5x5 conv, 32 inputs, 64 outputs
            'wconv4': tf.Variable(tf.random_normal([3, 3, 256, 256]), name='wconv4'),
            'bconv4': tf.Variable(tf.random_normal([256]), name='bconv4'),
            # 5x5 conv, 1 input, 32 outputs
            'wconv5': tf.Variable(tf.random_normal([3, 3, 256, 512]), name='wconv5'),
            'bconv5': tf.Variable(tf.random_normal([512]), name='bconv5'),
            # 5x5 conv, 32 inputs, 64 outputs
            'wconv6': tf.Variable(tf.random_normal([3, 3, 512, 512]), name='wconv6'),
            'bconv6': tf.Variable(tf.random_normal([512]), name='bconv6'),
            'wfc': tf.Variable(tf.random_normal([512,self.num_classes]), name='wfc1'),
            'bfc': tf.Variable(tf.random_normal([self.num_classes]), name='bfc1'),
            'wfc1': tf.Variable(tf.random_normal([512,1000]), name='wfc1'),
            'bfc1': tf.Variable(tf.random_normal([1000]), name='bfc1'),
            'wfc2': tf.Variable(tf.random_normal([1000,self.num_classes]), \
                name='wfc2'),
            'bfc2': tf.Variable(tf.random_normal([self.num_classes]), name='bfc2'),
            #'wfc3': tf.Variable(tf.random_normal([1024,1000]), name='wfc3'),
            #'bfc3': tf.Variable(tf.random_normal([1000]), name='bfc3'),
            #'wfc4': tf.Variable(tf.random_normal([1000,self.num_classes]), name='wfc4'),
            #'bfc4': tf.Variable(tf.random_normal([self.num_classes]), name='bfc4'),
        }
        # create the network graph
        self.images_placeholder = tf.placeholder(dtype=tf.float32, \
            shape=[None, None, None, 1])
        network = self.createCNNModel(self.images_placeholder)
        self.containedClassesPrediction = self.createFullyConvolutional(network)
        self.classes = tf.sigmoid(self.containedClassesPrediction)
        network, self.num_features = self.createEncoderLSTM(network)
        self.network = self.createDecoderLSTM(network)
        #
        #self.images_placeholder2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 512])
        #self.encoded = self.createEncoderLSTM(self.images_placeholder2)
        #self.containedClassesPredictionRefined = self.createFullyConvolutional2(self.encoded)
        #self.classesRefined = tf.sigmoid(self.encoded)
        #
        self.saver = tf.train.Saver()

    def createCNNModel(self, network):
        network = tf.nn.conv2d(network, self.weights['wconv1'], strides=[1,1,1,1], \
            padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv1'])
        network = max_pool_2d(network, 2, strides=2)
        network = tf.nn.relu(network)

        network = tf.nn.conv2d(network, self.weights['wconv2'], strides=[1,1,1,1], \
            padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv2'])
        network = max_pool_2d(network, 2, strides=2)
        network = tf.nn.relu(network)

        network = tf.nn.conv2d(network, self.weights['wconv3'], strides=[1,1,1,1], \
            padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv3'])
        network = tflearn.layers.normalization.batch_normalization(network) #same as torch?
        network = tf.nn.relu(network)

        network = tf.nn.conv2d(network, self.weights['wconv4'], strides=[1,1,1,1], \
            padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv4'])
        network = max_pool_2d(network, [2,1], strides=[2,1])
        network = tf.nn.relu(network)

        network = tf.nn.conv2d(network, self.weights['wconv5'], strides=[1,1,1,1], \
            padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv5'])
        network = tflearn.layers.normalization.batch_normalization(network) #same as torch?
        network = max_pool_2d(network, [1,2], strides=[1,2])
        network = tf.nn.relu(network)

        network = tf.nn.conv2d(network, self.weights['wconv6'], strides=[1,1,1,1], \
            padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv6'])
        network = tflearn.layers.normalization.batch_normalization(network)
        network = tf.nn.relu(network)
        return network

    def createFullyConvolutional(self, network):
        network = tflearn.layers.conv.global_avg_pool(network)
        #network = tf.tensordot(network,self.weights['wfc'],[[1],[0]])
        #network = network + self.weights['bfc']
        #code.interact(local=locals())
        network = tf.tensordot(network,self.weights['wfc1'],[[1],[0]])
        network = network + self.weights['bfc1']
        network = tf.nn.relu(network)
        network = tf.tensordot(network,self.weights['wfc2'],[[1],[0]])
        network = network + self.weights['bfc2']
        return network

    def createFullyConvolutional2(self, network):
        network = tflearn.layers.conv.global_avg_pool(network)
        #network = tf.tensordot(network,self.weights['wfc'],[[1],[0]])
        #network = network + self.weights['bfc']
        #code.interact(local=locals())
        network = tf.tensordot(network,self.weights['wfc3'],[[1],[0]])
        network = network + self.weights['bfc3']
        network = tf.nn.relu(network)
        network = tf.tensordot(network,self.weights['wfc4'],[[1],[0]])
        network = network + self.weights['bfc4']
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
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(rnncell_fw, \
                rnncell_bw, network[i], initial_state_fw=fw_state, \
                initial_state_bw=bw_state)
            fw_state, bw_state = output_states
            # code.interact(local=locals())
            l = l.write(i, outputs)        
            return [tf.add(i, 1), network, l, fw_state, bw_state]
        i, network, l, fw_state, bw_state = tf.while_loop(while_condition, body, \
            params)
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
        shape = tf.Print(tf.shape(network),[tf.shape(network)], \
            "dynamic network shape from decoder input!!!!!!",1)
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
        t, network, l, h, y, o, beta, network_, w1, \
            w2 = tf.while_loop(while_condition, body, params)
        #t = tf.Print(t,[t],"ttttttttttttttttttttttttttttend =============================",5)
        l = l.stack()
        l = tf.transpose(l, [1,0,2])
        #l = tf.Print(l,[tf.shape(l)],"after decoder: ")
        return l

    def loss(self, pred, labels):
        pred = tf.transpose(pred, [1,0,2])
        labels = tf.transpose(labels, [1,0])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, \
            logits=pred)
        return tf.reduce_mean(loss)

    def containedClassesLoss(self, pred, labels):
        #code.interact(local=locals())
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=pred)
        return tf.reduce_mean(loss)

    def training(self, loss):
        optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9)
        train_op = optimizer.minimize(loss)
        return train_op

    def addLog(self, train_loss, val_loss, train_accuracy, val_accuracy, epoch, \
            batch):
        with open(self.logs+'/log_'+str(epoch)+'_'+str(batch)+'.npz', 'w') \
                as fout:
            np.savez(fout,train_loss=train_loss,val_loss=val_loss, \
                train_accuracy=train_accuracy,val_accuracy=val_accuracy)

    def save(self, sess):
        shutil.rmtree(self.param_path, ignore_errors=True)
        if not os.path.exists(self.param_path):
            with open(self.param_path, 'w') as fout:
                np.savez(fout, num_classes=self.num_classes, \
                    max_num_tokens=self.max_num_tokens, \
                    capacity=self.capacity, \
                    learning_rate=self.learning_rate, \
                    num_features=self.num_features, \
                    nr_epochs=self.nr_epochs, model_dir=self.model_dir)
        self.saver.save(sess, self.save_path)

    def countVariables(self,sess):
        return np.sum([np.prod(v.get_shape().as_list()) \
            for v in tf.trainable_variables()])

def load(model_path, sess):
    params = np.load(model_path + '/params.npz')
    num_classes = np.asscalar(params['num_classes'])
    max_num_tokens = np.asscalar(params['max_num_tokens'])
    model_dir = np.asscalar(params['model_dir'])
    capacity = np.asscalar(params['capacity'])
    learning_rate = np.asscalar(params['learning_rate'])
    num_features = np.asscalar(params['num_features'])
    nr_epochs = np.asscalar(params['nr_epochs'])
    #code.interact(local=dict(globals(), **locals())) 
    model = Model(num_classes, max_num_tokens, model_dir, capacity, learning_rate, \
        num_features, nr_epochs)
    model.saver.restore(sess, model.save_path)
    return model
