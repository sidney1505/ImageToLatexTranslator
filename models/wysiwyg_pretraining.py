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
    def __init__(self,  num_classes, max_num_tokens, minibatchsize=50000, learning_rate=0.1, num_features=512, nr_epochs=50, model_dir=''):
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
            # 5x5 conv, 32 inputs, 64 outputs
            'wfc': tf.Variable(tf.random_normal([512,self.num_classes]), name='wfc'),
            'bfc': tf.Variable(tf.random_normal([self.num_classes]), name='bfc'),
        }


    def createCNNModel(self, network):
        network = tf.nn.conv_2d(network, self.weights['wconv1'], strides=[1,1,1,1], padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv1'])
        network = tf.nn.max_pool_2d(network, 2, strides=2, padding=2)
        network = tf.nn.relu(network)

        network = tf.nn.conv_2d(network, self.weights['wconv2'], strides=[1,1,1,1], padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv2'])
        network = tf.nn.max_pool_2d(network, 2, strides=2, padding=0)
        network = tf.nn.relu(network)

        network = tf.nn.conv_2d(network, self.weights['wconv3'], strides=[1,1,1,1], padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv3'])
        network = tflearn.layers.normalization.batch_normalization(network) #same as torch?
        network = tf.nn.relu(network)

        network = tf.nn.conv_2d(network, self.weights['wconv4'], strides=[1,1,1,1], padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv4'])
        network = tf.nn.max_pool_2d(network, [2,1], strides=[2,1])
        network = tf.nn.relu(network)

        network = tf.nn.conv_2d(network, self.weights['wconv5'], strides=[1,1,1,1], padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv5'])
        network = tflearn.layers.normalization.batch_normalization(network) #same as torch?
        network = tf.nn.max_pool_2d(network, [1,2], strides=[1,2])
        network = tf.nn.relu(network)

        network = tf.nn.conv_2d(network, self.weights['wconv6'], strides=[1,1,1,1], padding='SAME')
        network = tf.nn.bias_add(network, self.weights['bconv6'])
        network = tflearn.layers.normalization.batch_normalization(network)
        network = tf.nn.relu(network)

        network = tf.nn.max_pool_2d(network, [network.shape[1],network.shape[2]])
        network = tf.squeeze(network)
        network = tf.tensordot(network,self.weights['wfc'],[[1],[0]])
        network = network + self.weights['wfb']
        return network

    def loss(self, pred, labels):
        pred = tf.transpose(pred, [1,0,2])
        labels = tf.transpose(labels, [1,0])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=pred)
        return tf.reduce_mean(loss)

    def training(self, loss):
        optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9)
        train_op = optimizer.minimize(loss) #, global_step=global_step)
        return train_op

    def addTrainLog(self, loss_value, epoch, batch):
        with open(self.train_logs+'/log_'+str(epoch)+'_'+str(batch)+'.npz', 'w') as fout:
            np.savez(fout,val=np.mean(loss_value))

    def addValidationLog(self, loss_value, epoch, batch):
        with open(self.validation_logs+'/log_'+str(epoch)+'_'+str(batch)+'.npz', 'w') as fout:
            np.savez(fout,val=np.mean(loss_value))

    def save(self, sess):
        shutil.rmtree(self.param_path, ignore_errors=True)
        if not os.path.exists(self.param_path):
            with open(self.param_path, 'w') as fout:
                np.savez(fout, num_classes=self.num_classes, max_num_tokens=self.max_num_tokens, minibatchsize=self.minibatchsize, learning_rate=self.learning_rate, num_features=self.num_features, nr_epochs=self.nr_epochs, model_dir=self.model_dir)
        self.saver.save(sess, self.save_path)

def load(model_path, sess):
    params = np.load(model_path + '/params.npz')
    num_classes = np.asscalar(params['num_classes'])
    max_num_tokens = np.asscalar(params['max_num_tokens'])
    minibatchsize = np.asscalar(params['minibatchsize'])
    learning_rate = np.asscalar(params['learning_rate'])
    num_features = np.asscalar(params['num_features'])
    nr_epochs = np.asscalar(params['nr_epochs'])
    model_dir = np.asscalar(params['model_dir'])
    model = Model(num_classes, max_num_tokens, minibatchsize, learning_rate, num_features, nr_epochs, model_dir)
    model.saver.restore(sess, model.save_path)
    return model
