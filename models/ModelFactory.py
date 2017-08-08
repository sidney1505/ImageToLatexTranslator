import code, os, shutil, tflearn
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
import WYSIWYGCNN, BaselineEncoder, BaselineDecoder, SimpleAttentionModule, FullyConnected
import WYSIWYGEncoder, WYSIWYGDecoder, VGGNet, FullyConnectedVGG, LibraryDecoder

class Model:
    def __init__(self,  num_classes, max_num_tokens, model_dir, vocabulary, \
            capacity=30000, learning_rate=0.1, nr_epochs=50, train_mode=1, loaded=False, \
            session=None):
        # intialise class variables
        self.num_classes = num_classes
        self.vocabulary = np.array(vocabulary)
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
        self.save_path = self.model_dir + '/weights.ckpt'
        self.param_path = self.model_dir + '/params.npz'
        self.logfile_path = self.model_dir + '/logfile'
        if not os.path.exists(self.logfile_path):
            os.makedirs(self.logfile_path)
        if not os.path.exists(self.param_path):
            with open(self.param_path, 'w') as fout:
                np.savez(fout, num_classes=num_classes, \
                    capacity=capacity, learning_rate=learning_rate, \
                    nr_epochs=nr_epochs, vocabulary=self.vocabulary,\
                    model_dir=model_dir)
        if session == None:
            self.session = tf.Session()
        else:
            self.session = session
        # create the network graph depending on train mode
        self.weights = {}
        self.classes_prediction = tf.constant(-1)
        self.classes_gold = None
        if train_mode == 0:
            print('build model type 0!')
            # end-to-end training like the original wysiwyg
            # whole pipeline encoding -> refinement -> decoding
            # very slow and space consuming, so care about the capacity
            self.input = tf.placeholder(dtype=tf.float32, \
                shape=[None, None, None, 1])
            self.label_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])
            self.classes_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,num_classes])
            self.features = WYSIWYGCNN.createCNNModel(self, self.input, loaded)
            self.featuresRefined = WYSIWYGEncoder.createEncoderLSTM(self, self.features)
            self.label_prediction = WYSIWYGDecoder.createDecoderLSTM(self, \
                self.featuresRefined)            
            self.classes_prediction = tf.sigmoid(FullyConnected.createFullyConnected(self, \
                self.features, loaded))
            self.useLabelLoss()
        elif train_mode == 1:
            print('build model type 1!')
            # seperate encoding of the images to a feature map
            self.input = tf.placeholder(dtype=tf.float32, \
                shape=[None, None, None, 1])
            self.classes_gold = tf.placeholder(dtype=tf.float32, \
                shape=[None,num_classes])
            self.features = WYSIWYGCNN.createCNNModel(self, self.input, loaded)           
            self.classes_prediction = FullyConnected.createFullyConnected(self, \
                self.features, loaded)
            self.useClassesLoss()
        elif train_mode == 2:
            print('build model type 2!')
            # seperate refinement of the feature map
            self.input = tf.placeholder(dtype=tf.float32, shape= \
                [None, None, None, 512])
            self.classes_gold = tf.placeholder(dtype=tf.float32, \
                shape=[None,num_classes])
            self.featuresRefined = WYSIWYGEncoder.createEncoderLSTM(self, self.input)
            self.classes_prediction = FullyConnected.createFullyConnected(self, \
                self.featuresRefined, loaded)
            self.useClassesLoss()
        elif train_mode == 3:
            print('build model type 3!')
            # seperate decoding of the feature map
            self.input = tf.placeholder(dtype=tf.float32, shape= \
                [None, None, None, 512])
            self.label_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])
            self.label_prediction = WYSIWYGDecoder.createDecoderLSTM(self, self.input)
            self.useLabelLoss()
        elif train_mode == 4:
            print('build model type 4!')
            # a more simple attention module
            self.input = tf.placeholder(dtype=tf.float32, shape= \
                [None, None, None, 512])
            self.label_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])
            self.label_prediction = SimpleAttentionModule.createSimpleAttentionModule(self, \
                self.input)
            self.useLabelLoss()
        elif train_mode == 5:
            print('build model type 5!')
            # the basic attention module
            self.input = tf.placeholder(dtype=tf.float32, shape= \
                [None, None, None, 512])
            self.label_gold = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])            
            outputs, self.encoded = BaselineEncoder.createBaselineEncoder(self, \
                self.input)
            self.label_prediction = BaselineDecoder.createBaselineDecoder(self, self.encoded)
            #print('in init 5')
            #code.interact(local=dict(globals(), **locals()))
            self.useLabelLoss()
        elif train_mode == 6:
            print('build model type 6!')
            # seperate encoding of the images to a feature map
            self.input = tf.placeholder(dtype=tf.float32, \
                shape=[None, None, None, 1])
            self.classes_gold = tf.placeholder(dtype=tf.float32, \
                shape=[None,num_classes])
            self.features = VGGNet.createCNNModel(self, self.input, loaded)
            self.classes_prediction = FullyConnectedVGG.createFullyConnected(self, \
                self.features, loaded)
            self.useClassesLoss()
        elif train_mode == 7:
            print('build model type 7!')
            # the basic attention module
            self.input = tf.placeholder(dtype=tf.float32, shape= [None, None, None, 512])
            self.label_gold = tf.placeholder(dtype=tf.int32, shape=[None,max_num_tokens])            
            outputs, state = BaselineEncoder.createBaselineEncoder(self, self.input)
            self.label_prediction = LibraryDecoder.createAttentionModule(self, outputs, \
                state)
            self.useLabelLoss()
        assert self.input != None
        assert self.loss != None
        self.useAdadeltaOptimizer()
        # initialises all variables
        if not loaded:
            #code.interact(local=dict(globals(), **locals()))
            init = tf.global_variables_initializer()
            self.session.run(init)
            for v in tf.trainable_variables():
                #code.interact(local=dict(globals(), **locals()))
                seed = np.random.standard_normal(v.shape.as_list()) * 0.05
                v = v.assign(seed)
                x = self.session.run(v)
            self.save()
        tf.summary.histogram('predictionHisto', self.predictionDistribution)
        tf.summary.scalar('predictionAvg', tf.reduce_mean(self.predictionDistribution))
        tf.summary.tensor_summary('predictionTensor', self.predictionDistribution)
        self.summaries = tf.summary.merge_all()
        self.board_path = self.model_dir + '/tensorboard'
        self.writer = tf.summary.FileWriter(self.board_path, graph=tf.get_default_graph())
        self.step = 0
        self.predictionDistributionsDone = 0      

    # indicates to fit the predicted label to the gold label as objective
    def useLabelLoss(self):
        with tf.variable_scope("labelLoss", reuse=None):
            self.groundtruth = self.label_gold
            label_prediction = tf.transpose(self.label_prediction, [1,0,2])
            label_gold = tf.transpose(self.label_gold, [1,0])
            #code.interact(local=dict(globals(), **locals())) 
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_gold, \
                logits=label_prediction)
            self.loss = tf.reduce_mean(loss)
            self.predictionDistribution = tf.nn.softmax(self.label_prediction)
            self.prediction = tf.argmax(self.predictionDistribution, axis=2)
            self.used_loss = 'label'

    # indicates to fit the predicted classes to the gold classes as objective
    def useClassesLoss(self):
        with tf.variable_scope("classesLoss", reuse=None):
            self.groundtruth = self.classes_gold
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.classes_gold, \
                logits=self.classes_prediction)
            self.loss = tf.reduce_mean(loss)
            self.classes_prediction = tf.sigmoid(self.classes_prediction)
            self.predictionDistribution = self.classes_prediction
            self.prediction = tf.round(self.predictionDistribution)
            self.used_loss = 'classes'

    def countBatchDuplicates(self, tensor):
        params = [tf.constant(0), tf.constant(0), tensor]
        def while_condition(batch, counter, tensor):
            return tf.less(batch,tf.shape(tensor)[0])
        def body(batch, counter, tensor):
            return [batch, counter + self.countDuplicates(tensor[batch]), tensor]
        _,counter,_ = tf.while_loop(while_condition, body, params)
        return counter

    def countDuplicates(self, tensor):
        counter = tf.constant(0)
        for vocab in range(len(self.vocabulary)):
            counter = counter + tf.maximum(tf.constant(0),self.tf_count(tensor, \
                tf.constant(vocab)) - tf.constant(1))
        return counter
            
    def tf_count(self, t, val):
        elements_equal_to_value = tf.equal(tf.cast(t, tf.int32), tf.cast(val,tf.int32))
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        return count

    # indicates to use the momentum optimizer
    def useGradientDescentOptimizer(self):
        with tf.variable_scope("MySGDOptimizer", reuse=None):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            #self.train_op = optimizer.minimize(self.loss)
            self.train_op = slim.learning.create_train_op(self.loss, optimizer, \
                summarize_gradients=True)

    # indicates to use the momentum optimizer
    def useMomentumOptimizer(self):
        with tf.variable_scope("MyMomentumOptimizer", reuse=None):
            optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9)
            #code.interact(local=dict(globals(), **locals()))
            self.train_op = slim.learning.create_train_op(self.loss, optimizer, \
                summarize_gradients=True)

    # indicates to use the momentum optimizer
    def useAdamOptimizer(self):
        with tf.variable_scope("MyAdamOptimizer", reuse=None):
            optimizer = tf.train.AdamOptimizer()
            #code.interact(local=dict(globals(), **locals()))
            self.train_op = slim.learning.create_train_op(self.loss, optimizer, \
                summarize_gradients=True)

    # indicates to use the momentum optimizer
    def useAdadeltaOptimizer(self):
        with tf.variable_scope("MyAdadeltaOptimizer", reuse=None):
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            #code.interact(local=dict(globals(), **locals()))
            self.train_op = slim.learning.create_train_op(self.loss, optimizer, \
                summarize_gradients=True)

    def addLog(self, train_loss, val_loss, train_accuracy, val_accuracy, epoch, batch):
        with open(self.logs+'/log_'+str(epoch)+'_'+str(batch)+'.npz', 'w') \
                as fout:
            np.savez(fout,train_loss=train_loss,val_loss=val_loss, \
                train_accuracy=train_accuracy,val_accuracy=val_accuracy)

    def trainStep(self, inp, groundtruth):
        if self.step % 10 == 0:
            feed_dict={self.input: inp, self.groundtruth: groundtruth}
            _,lossValue,summs,dis,pred = self.session.run([self.train_op, \
                self.loss, self.summaries, \
                self.predictionDistribution, self.prediction], feed_dict=feed_dict)
            self.writer.add_summary(summs, global_step=self.step)
            self.writeInLogfile(pred, dis, groundtruth)
            print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
            print('Current Weights:')
            self.printTrainableVariables()
            print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
            if self.step % 50 == 0:
                print('#############################################################')
                print('#############################################################')
                print('#############################################################')
                print('###                      Save model!                      ###')
                print('#############################################################')
                print('#############################################################')
                print('#############################################################')
                self.save()
        else:
            feed_dict={self.input: inp, self.groundtruth: groundtruth}
            _,lossValue, pred = self.session.run([self.train_op, \
                self.loss, self.prediction], feed_dict=feed_dict)
        self.step = self.step + 1
        return lossValue, self.calculateAccuracy(pred, groundtruth)

    def calculateAccuracy(self, pred, gt):
        matches = 0
        count = 0
        for batch in range(pred.shape[0]):
            for token in range(pred.shape[1]):
                count = count + 1
                if pred[batch][token] == gt[batch][token]:
                    matches = matches + 1
        return float(matches) / float(count)

    def valStep(self, inp, groundtruth):
        feed_dict={self.input: inp, self.groundtruth: groundtruth}
        lossValue, dis, pred = self.session.run([self.loss, \
            self.predictionDistribution, self.prediction], feed_dict=feed_dict)
        self.writeInLogfile(pred, dis, groundtruth)
        return lossValue, self.calculateAccuracy(pred, groundtruth)

    def writeInLogfile(self, prediction, prediction_distribution, groundtruth):
        if self.used_loss == 'label':
            #code.interact(local=dict(globals(), **locals()))
            for batch in range(prediction.shape[0]):
                predictionPath = self.logfile_path + '/prediction' + str(\
                    self.predictionDistributionsDone)
                fout = open(predictionPath, 'w')
                for token in range(prediction.shape[1]):
                    #code.interact(local=dict(globals(), **locals()))
                    line='\"'+self.vocabulary[prediction[batch][token]]+'\" : '\
                        + str(np.max(prediction_distribution[batch][token])) + ' -> \"' \
                        + self.vocabulary[int(groundtruth[batch][token])] + '\" : ' \
                        + str(prediction_distribution[batch][token] \
                        [int(groundtruth[batch][token])]) + '\n'
                    fout.write(line)
                fout.close()
                self.predictionDistributionsDone = self.predictionDistributionsDone + 1

    def predict(self, inp, gt):
        feed_dict={self.input: inp, self.label_gold: gt}
        #code.interact(local=dict(globals(), **locals()))
        return self.session.run(self.prediction, feed_dict=feed_dict)

    def save(self):
        shutil.rmtree(self.param_path, ignore_errors=True)
        #if not os.path.exists(self.param_path):
        with open(self.param_path, 'w') as fout:
            #print('in der save methode')
            #code.interact(local=dict(globals(), **locals()))
            np.savez(fout, num_classes=self.num_classes, \
                capacity=self.capacity, \
                train_mode=self.train_mode, \
                learning_rate=self.learning_rate, \
                nr_epochs=self.nr_epochs, 
                vocabulary=self.vocabulary,
                model_dir=self.model_dir)
        saver = tf.train.Saver()
        #print('in der save methode 2')
        #code.interact(local=dict(globals(), **locals()))
        saver.save(self.session, self.save_path)
        print('variables saved!')
        self.printGlobalVariables()
        #code.interact(local=dict(globals(), **locals()))

    def printTrainableVariables(self):
        for v in tf.trainable_variables():
            x = self.session.run(v)
            print(v.name + ' : ' + str(v.get_shape()) + ' : ' + str(np.mean(x)) + ' : ' \
                + str(np.var(x)))
            #print(x)

    def printGlobalVariables(self):
        for v in tf.global_variables():
            x = self.session.run(v)
            print(v.name + ' : ' + str(v.get_shape()) + ' : ' + str(np.mean(x)) + ' : ' \
                + str(np.var(x)))

    def countVariables(self,sess):
        return np.sum([np.prod(v.get_shape().as_list()) \
            for v in tf.trainable_variables()])

def load(model_dir, train_mode, max_num_tokens):
    params = np.load(model_dir + '/params.npz')
    #code.interact(local=dict(globals(), **locals()))
    num_classes = np.asscalar(params['num_classes'])
    vocabulary = np.asarray(params['vocabulary'])
    #model_dir = np.asscalar(params['model_dir'])
    capacity = np.asscalar(params['capacity'])
    learning_rate = np.asscalar(params['learning_rate'])
    nr_epochs = np.asscalar(params['nr_epochs'])
    print('try to restore model!')
    tf.reset_default_graph()
    session = tf.Session()
    model = Model(num_classes, max_num_tokens, model_dir, vocabulary, capacity, \
        learning_rate, nr_epochs, train_mode=train_mode, loaded=True, session=session)    
    saver = tf.train.Saver()
    print('load variables!')
    saver.restore(session, model_dir + '/weights.ckpt')
    model.printGlobalVariables()
    print('model restored!')
    code.interact(local=dict(globals(), **locals()))
    return model
