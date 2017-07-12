import code, os, shutil, tflearn
import tensorflow as tf
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
import WYSIWYGCNN, BaselineEncoder, BaselineDecoder, SimpleAttentionModule, FullyConnected
import WYSIWYGEncoder, WYSIWYGDecoder

class Model:
    def __init__(self,  num_classes, max_num_tokens, model_dir, \
            capacity=30000, learning_rate=0.001, nr_epochs=50, train_mode=1, loaded=False):
        # intialise class variables
        self.num_classes = num_classes
        self.max_num_tokens = max_num_tokens
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.nr_epochs = nr_epochs
        self.train_mode = train_mode
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
                    nr_epochs=nr_epochs, \
                    model_dir=model_dir)
        # create the network graph depending on train mode
        self.weights = {}
        self.classes_prediction = tf.constant(-1)
        self.classes_gold = None
        if train_mode == 0:            
            self.input = tf.placeholder(dtype=tf.float32, \
                shape=[None, None, None, 1])
            self.label_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])
            self.classes_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,num_classes])
            self.features = WYSIWYGCNN.createCNNModel(self, self.input)
            self.featuresRefined = WYSIWYGEncoder.createEncoderLSTM(self, self.features)
            self.label_prediction = WYSIWYGDecoder.createDecoderLSTM(self, \
                self.featuresRefined)            
            self.classes_prediction = tf.sigmoid(FullyConnected.createFullyConnected(self, \
                self.features))
            self.useLabelLoss()
        elif train_mode == 1:
            self.input = tf.placeholder(dtype=tf.float32, \
                shape=[None, None, None, 1])
            self.classes_gold = tf.placeholder(dtype=tf.float32, \
                shape=[None,num_classes])
            self.features = WYSIWYGCNN.createCNNModel(self, self.input)           
            self.classes_prediction = FullyConnected.createFullyConnected(self, \
                self.features)
            self.useClassesLoss()
        elif train_mode == 2:
            self.input = tf.placeholder(dtype=tf.float32, shape= \
                [None, None, None, 512])
            self.classes_gold = tf.placeholder(dtype=tf.float32, \
                shape=[None,num_classes])
            self.featuresRefined = WYSIWYGEncoder.createEncoderLSTM(self, self.input)
            self.classes_prediction = FullyConnected.createFullyConnected(self, \
                self.featuresRefined)
            self.useClassesLoss()
        elif train_mode == 3:
            self.input = tf.placeholder(dtype=tf.float32, shape= \
                [None, None, None, 512])
            self.label_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])
            self.label_prediction = WYSIWYGDecoder.createDecoderLSTM(self, self.input)
            self.useLabelLoss()
        elif train_mode == 4:
            self.input = tf.placeholder(dtype=tf.float32, shape= \
                [None, None, None, 512])
            self.label_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])
            self.label_prediction = SimpleAttentionModule.createSimpleAttentionModule(self, \
                self.input)
            self.useLabelLoss()
        elif train_mode == 5:
            self.input = tf.placeholder(dtype=tf.float32, shape= \
                [None, None, None, 512])
            self.label_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])            
            self.encoded = BaselineEncoder.createBaselineEncoder(self, \
                self.input)
            self.label_prediction = BaselineDecoder.createBaselineDecoder(self, self.encoded)
            self.useLabelLoss()
        assert self.input != None
        assert self.loss != None
        self.useMomentumOptimizer()
        #
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        # initialises all variables
        if not loaded:
            init = tf.global_variables_initializer()
            self.session.run(init)
        # intial save! save seeds??
        self.save()

    def useLabelLoss(self):
        self.groundtruth = self.label_gold
        label_prediction = tf.transpose(self.label_prediction, [1,0,2])
        label_gold = tf.transpose(self.label_gold, [1,0])
        #code.interact(local=dict(globals(), **locals())) 
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_gold, \
            logits=label_prediction)
        self.loss = tf.reduce_mean(loss)
        self.label_gold = tf.nn.softmax(self.label_prediction)
        self.prediction = self.label_prediction
        self.used_loss = 'label'

    def useClassesLoss(self):
        self.groundtruth = self.classes_gold
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.classes_gold, \
            logits=self.classes_prediction)
        self.loss = tf.reduce_mean(loss)
        self.classes_prediction = tf.sigmoid(self.classes_prediction)
        self.prediction = self.classes_prediction
        self.used_loss = 'classes'

    def useMomentumOptimizer(self):
        optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9)
        self.train_op = optimizer.minimize(self.loss)

    def addLog(self, train_loss, val_loss, train_accuracy, val_accuracy, epoch, \
            batch, train_stats, val_stats):
        with open(self.logs+'/log_'+str(epoch)+'_'+str(batch)+'.npz', 'w') \
                as fout:
            np.savez(fout,train_loss=train_loss,val_loss=val_loss, \
                train_accuracy=train_accuracy,val_accuracy=val_accuracy, \
                train_stats=train_stats,val_stats=val_stats)

    def trainStep(self, inp, groundtruth):
        feed_dict={self.input: inp, self.groundtruth: groundtruth}
        return self.session.run([self.train_op, self.loss, self.classes_prediction], \
                        feed_dict=feed_dict)

    def valStep(self, inp, groundtruth):
        feed_dict={self.input: inp, self.groundtruth: groundtruth}
        return self.session.run([self.loss, self.classes_prediction], \
                        feed_dict=feed_dict)

    def save(self):
        shutil.rmtree(self.param_path, ignore_errors=True)
        if not os.path.exists(self.param_path):
            with open(self.param_path, 'w') as fout:
                np.savez(fout, num_classes=self.num_classes, \
                    max_num_tokens=self.max_num_tokens, \
                    capacity=self.capacity, \
                    train_mode=train_mode, \
                    learning_rate=self.learning_rate, \
                    nr_epochs=self.nr_epochs, model_dir=self.model_dir)
        self.saver.save(self.session, self.save_path)

    def countVariables(self,sess):
        return np.sum([np.prod(v.get_shape().as_list()) \
            for v in tf.trainable_variables()])

def load(model_path, sess, train_mode):
    params = np.load(model_path + '/params.npz')
    num_classes = np.asscalar(params['num_classes'])
    max_num_tokens = np.asscalar(params['max_num_tokens'])
    model_dir = np.asscalar(params['model_dir'])
    capacity = np.asscalar(params['capacity'])
    learning_rate = np.asscalar(params['learning_rate'])
    nr_epochs = np.asscalar(params['nr_epochs'])
    #code.interact(local=dict(globals(), **locals()))
    model = Model(num_classes, max_num_tokens, model_dir, capacity, learning_rate, \
        nr_epochs, train_mode=train_mode, loaded=True)
    model.saver.restore(sess, model.save_path)
    return model
