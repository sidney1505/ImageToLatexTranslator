import code, os, shutil, tflearn, datetime, sys
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
# import feature extractors
from FeatureExtractors.WYSIWYGFeatureExtractor import WYSIWYGFeatureExtractor
from FeatureExtractors.AlexnetFeatureExtractor import AlexnetFeatureExtractor
from FeatureExtractors.VGGFeatureExtractor import VGGFeatureExtractor
from FeatureExtractors.ResnetFeatureExtractor import ResnetFeatureExtractor
from FeatureExtractors.DensenetFeatureExtractor import DensenetFeatureExtractor
# import classifiers
from Classifiers.SimpleClassifier import SimpleClassifier
# import encoders
from Encoders.MonorowEncoder import MonorowEncoder
from Encoders.BirowEncoder import BirowEncoder
from Encoders.MonocolEncoder import MonocolEncoder
from Encoders.BicolEncoder import BicolEncoder
from Encoders.QuadroEncoder import QuadroEncoder
# import decoders
from Decoders.SimpleDecoder import SimpleDecoder
from Decoders.SimplegruDecoder import SimplegruDecoder
from Decoders.BahdanauDecoder import BahdanauDecoder
from Decoders.LuongDecoder import LuongDecoder

class Model:
    # train_mode????
    def __init__(self, model_dir, feature_extractor='', encoder='',\
            decoder='', encoder_size=-1, decoder_size=-1, optimizer='', \
            learning_rate=0, vocabulary='', max_num_tokens=None, \
            capacity=-1, session=None, loaded=False):
        tf.reset_default_graph()
        # temporory solution
        self.num_features = 512
        #
        if loaded:
            # load parameters from given directory
            self.model_dir = model_dir
            # unchangable parameters
            self.feature_extractor = self.readParam('feature_extractor')
            self.encoder = self.readParam('encoder')
            self.decoder = self.readParam('decoder')
            self.encoder_size = int(self.readParam('encoder_size'))
            self.decoder_size = int(self.readParam('decoder_size'))
            self.vocabulary = np.array(self.readParam('vocabulary').split('\n'))
            # changable parameters
            self.optimizer = self.readParam('optimzer')
            self.learning_rate = self.readParam('learning_rate')
            self.max_num_tokens = int(self.readParam('max_num_tokens'))
            self.capacity = int(self.readParam('capacity'))
            # implicit parameters
            self.current_epoch = int(self.readParam('current_epoch'))
            self.step = int(self.readParam('current_epoch'))
            self.current_millis = int(self.readParam('current_millis'))
        else:
            # create directory for the model and store the parameters there
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_dir = model_dir + '/' + feature_extractor + '_' + encoder + '_' + \
                decoder + '_' + str(encoder_size) + '_' + str(decoder_size) + '_' + \
                optimizer + '_' + str(datetime.datetime.now())
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            self.model_dir = model_dir
            self.writeParam('model_dir',model_dir)
            self.feature_extractor = feature_extractor
            self.writeParam('feature_extractor',feature_extractor)
            self.encoder = encoder
            self.writeParam('encoder',encoder)
            self.decoder = decoder
            self.writeParam('decoder',decoder)
            self.encoder_size = encoder_size
            self.writeParam('encoder_size',str(encoder_size))
            self.decoder_size = decoder_size
            self.writeParam('decoder_size',str(decoder_size))
            self.vocabulary = np.array(vocabulary.split('\n'))
            self.writeParam('vocabulary',str(vocabulary))
            self.optimizer = optimizer
            self.writeParam('optimizer',str(optimizer))
            self.learning_rate = learning_rate
            self.writeParam('learning_rate',str(learning_rate))
            self.max_num_tokens = max_num_tokens
            self.writeParam('max_num_tokens',str(max_num_tokens))
            self.capacity = capacity
            self.writeParam('capacity',str(capacity))
            #
            self.current_epoch = 0
            self.current_step = 0
            self.current_millis = 0
        self.num_classes = len(self.vocabulary)
        # create necessary paths
        self.logs = self.model_dir + '/logs'
        if not os.path.exists(self.logs):
            os.makedirs(self.logs)
        self.logfile_dir = self.model_dir + '/logfile'
        if not os.path.exists(self.logfile_dir):
            os.makedirs(self.logfile_dir)
        if session == None:
            self.session = tf.Session()
        else:
            self.session = session
        # placeholders which are necessary to seperate phases
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, [])
        # building the network graph
        if self.feature_extractor != '':
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
            if self.feature_extractor == 'wysiwygFe':
                self.features = WYSIWYGFeatureExtractor(self).createGraph()
            elif self.feature_extractor == 'alexnetFe':
                self.features = AlexnetFeatureExtractor(self).createGraph()
            elif self.feature_extractor == 'vggFe':
                self.features = VGGFeatureExtractor(self).createGraph()
            elif self.feature_extractor == 'resnetFe':
                self.features = ResnetFeatureExtractor(self).createGraph()
            elif self.feature_extractor == 'densenetFe':
                self.features =DensenetFeatureExtractor(self).createGraph()
            else:
                print(self.feature_extractor + ' is no valid feature extractor!')
                quit()
        else:
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, \
                self.num_features])
            self.features = self.input
        if self.encoder != '' and self.decoder != '' and self.encoder_size != -1 and \
                self.decoder_size != -1:
            self.groundtruth = tf.placeholder(dtype=tf.int32, shape=[None, \
                self.max_num_tokens])
            # define the encoder
            if self.encoder == 'monorowEnc':
                self.outputs, self.state = MonorowEncoder(self).createGraph()
            elif self.encoder == 'birowEnc':
                self.outputs, self.state = BirowEncoder(self).createGraph()
            elif self.encoder == 'monocolEnc':
                self.outputs, self.state = MonocolEncoder(self).createGraph()
            elif self.encoder == 'bicolEnc':
                self.outputs, self.state = BicolEncoder(self).createGraph()
            elif self.encoder == 'quadroEnc':
                self.outputs, self.state = QuadroEncoder(self).createGraph()
            else:
                print(self.encoder + ' is no valid encoder type!')
                quit()
            # define the decoder
            if self.decoder == 'simpleDec':
                SimpleDecoder(self).createGraph()
            elif self.decoder == 'simplegruDec':
                SimplegruDecoder(self).createGraph()
            elif self.decoder == 'bahdanauDec':
                BahdanauDecoder(self).createGraph()
            elif self.decoder == 'luongDec':
                LuongDecoder(self).createGraph()
            else:
                print(self.decoder + ' is no valid decoder type!')
                quit()
            self.__useLabelLoss()
        elif self.encoder != '' or self.decoder != '' or self.encoder_size != -1 or \
                self.decoder_size != -1:
            print('encoder and decoder must be used together!')
            quit()
        else:
            self.prediction = SimpleClassifier(self).createGraph()
            self.__useClassesLoss()
        # creates the optimizer
        self.createOptimizer()
        #print('model factory')
        #code.interact(local=dict(globals(), **locals()))
        # create the save path
        self.save_path = self.model_dir + '/weights.ckpt'
        self.seed_path = self.model_dir + '/seeds.ckpt'
        # initialises all variables
        if not loaded:
            init = tf.global_variables_initializer()
            self.session.run(init)
            for v in tf.trainable_variables():
                #code.interact(local=dict(globals(), **locals()))
                seed = np.random.standard_normal(v.shape.as_list()) * 0.05
                v = v.assign(seed)
                x = self.session.run(v)
                #print(seed.shape)
            self.save()
            self.saveSeeds()
        #tf.summary.histogram('predictionHisto', self.prediction)
        #tf.summary.scalar('predictionAvg', tf.reduce_mean(self.prediction))
        #tf.summary.tensor_summary('predictionTensor', self.prediction)
        #self.summaries = tf.summary.merge_all()
        #self.board_path = self.model_dir + '/tensorboard'
        #self.writer = tf.summary.FileWriter(self.board_path, graph=tf.get_default_graph())
        # writes the parameter count
        self.writeParam('param_count', str(self.countVariables()))

    def decayLearningRate(self, decay):
        self.learning_rate = self.learning_rate * decay
        self.writeParam('learning_rate', self.learning_rate)
        self.setOptimizer()

    def createOptimizer(self):
        if self.optimizer == 'momentum':
            with tf.variable_scope("MyMomentumOptimizer", reuse=None):
                optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9)                
        elif self.optimizer == 'sgd':
            with tf.variable_scope("MySGDOptimizer", reuse=None):
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == 'adam':
            with tf.variable_scope("MyAdamOptimizer", reuse=None):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer == 'adadelta':
            with tf.variable_scope("MyAdadeltaOptimizer", reuse=None):
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 2.0) # in [1,5]
        self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

    # indicates to fit the predicted label to the gold label as objective
    def __useLabelLoss(self):
        try:
            self.used_loss = 'label'
            with tf.variable_scope("labelLoss", reuse=None):
                batchsize = tf.shape(self.train_prediction)
                self.train_distribution = tf.nn.softmax(self.train_prediction)
                self.train_pred = tf.argmax(self.train_distribution, axis=2)
                diff = self.max_num_tokens - tf.shape(self.train_pred)[1]
                self.prediction_greedy = tf.pad(self.train_pred, [[0,0],[0,diff]])
                eos_gt = tf.argmax(self.groundtruth, 1)
                eos_pred = tf.argmax(self.train_pred, 1)
                max_seq_length = tf.maximum(eos_gt, eos_pred)
                weights = tf.sequence_mask(max_seq_length, self.max_num_tokens)
                weights = tf.cast(weights, tf.float32)
                self.train_loss = tf.contrib.seq2seq.sequence_loss(self.train_prediction,\
                    self.groundtruth, weights)
                self.infer_distribution = tf.nn.softmax(self.infer_prediction)
                self.infer_pred = tf.argmax(self.infer_distribution, axis=2)
                diff = self.max_num_tokens - tf.shape(self.infer_pred)[1]
                self.prediction_greedy = tf.pad(self.infer_pred, [[0,0],[0,diff]])
                eos_gt = tf.argmax(self.groundtruth, 1)
                eos_pred = tf.argmax(self.infer_pred, 1)
                max_seq_length = tf.maximum(eos_gt, eos_pred)
                weights = tf.sequence_mask(max_seq_length, self.max_num_tokens)
                weights = tf.cast(weights, tf.float32)
                self.infer_loss = tf.contrib.seq2seq.sequence_loss(self.infer_prediction,\
                    self.groundtruth, weights)
        except Exception:
            print('exception in label loss')
            print(sys.exc_info())
            code.interact(local=dict(globals(), **locals()))

    # indicates to fit the predicted classes to the gold classes as objective
    def __useClassesLoss(self):
        self.used_loss = 'classes'
        with tf.variable_scope("classesLoss", reuse=None):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.groundtruth, \
                logits=self.prediction)
            self.loss = tf.reduce_mean(loss)
            self.prediction = tf.sigmoid(self.prediction)
            self.predictionDistribution = self.prediction
            self.prediction = tf.round(self.predictionDistribution)

    def trainStep(self, inp, groundtruth):
        feed_dict={self.input: inp, self.groundtruth: groundtruth, self.is_training:True, \
            self.keep_prob:0.5}
        _,self.current_train_loss, self.current_infer_loss, self.current_train_prediction, \
            self.current_infer_prediction = self.session.run([\
            self.update_step, self.train_loss, self.infer_loss, self.train_prediction, \
            self.infer_prediction], feed_dict=feed_dict)
        if self.current_step % 100 == 0:
            print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
            print('global step: ' + str(self.current_step))
            print('Current Weights:')
            self.printTrainableVariables()
            print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
            if self.current_step % 1000 == 0:
                self.save()
        self.current_train_accuracy = self.calculateAccuracy( \
            self.__argmaxs(self.current_train_prediction), groundtruth)
        infer_prediction = self.__argmaxs(self.current_infer_prediction)
        infer_prediction = self.fillWithEndTokens(infer_prediction)
        infer_prediction = infer_prediction.astype(int)
        self.current_infer_accuracy = self.calculateAccuracy( \
            infer_prediction, groundtruth)
        self.current_step  = self.current_step + 1

    def valStep(self, inp, groundtruth):
        try:
            #feed_dict={self.input: inp, self.groundtruth:np.ones(groundtruth.shape),\
            #    self.is_training:False, self.keep_prob:1.0}
            feed_dict={self.input: inp, self.groundtruth:groundtruth,\
                self.is_training:False, self.keep_prob:1.0}
            train_prediction, infer_prediction, lossValue, dis, pred = \
                self.session.run([self.train_prediction, self.infer_prediction, self.loss,\
                self.predictionDistribution, self.prediction], feed_dict=feed_dict)
            self.current_train_accuracy = self.calculateAccuracy( \
                self.__argmaxs(train_prediction), groundtruth)
            infer_prediction = self.__argmaxs(infer_prediction)
            infer_prediction = self.fillWithEndTokens(infer_prediction)
            infer_prediction = infer_prediction.astype(int)
            self.current_infer_accuracy = self.calculateAccuracy( \
                infer_prediction, groundtruth)
            self.writeInLogfile(pred, dis, groundtruth)
            return lossValue, self.current_infer_accuracy, infer_prediction
        except Exception:
            print('validation step failed!')
            code.interact(local=dict(globals(), **locals()))
            return 100, 0.0, np.zeros(groundtruth.shape)

    def fillWithEndTokens(self, prediction):
        prediction_filled = np.zeros([prediction.shape[0], self.max_num_tokens])
        for batch in range(prediction_filled.shape[0]):
            for token in range(prediction_filled.shape[1]):
                if token >= prediction.shape[1]:
                    prediction_filled[batch][token] = self.num_classes - 1
                else:
                    prediction_filled[batch][token] = prediction[batch][token]
        return prediction_filled

    def __argmaxs(self, distribution):
        __argmaxss = np.zeros([distribution.shape[0],distribution.shape[1]])
        for i in range(distribution.shape[0]):
            for j in range(distribution.shape[1]):
                __argmaxss[i][j] = np.argmax(distribution[i][j])
        return __argmaxss

    def predict(self, inp, gt):
        feed_dict={self.input: inp,  self.is_training:False, self.keep_prob:1.0,
            self.groundtruth:np.ones(gt.shape)}
        print('predict')
        #code.interact(local=dict(globals(), **locals()))
        return self.session.run(self.prediction, feed_dict=feed_dict)

    def writeInLogfile(self, prediction, prediction_distribution, groundtruth):
        if self.used_loss == 'label':
            #code.interact(local=dict(globals(), **locals()))
            for batch in range(prediction.shape[0]):
                predictionPath = self.logfile_dir + '/prediction' + str(\
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

    def addLog(self, train_loss, val_loss, train_accuracy, val_accuracy, epoch):
        with open(self.logs+'/log_'+str(epoch)+'.npz', 'w') as fout:
            np.savez(fout,train_loss=train_loss,val_loss=val_loss, \
                train_accuracy=train_accuracy,val_accuracy=val_accuracy)

    def calculateAccuracy(self, pred, gt):
        accuracies = []
        for batch in range(gt.shape[0]):
            matches = 0
            count = 0
            for token in range(gt.shape[1]):
                if pred[batch][token] == self.max_num_tokens - 1 and \
                        gt[batch][token] == self.max_num_tokens - 1:
                    break
                count = count + 1
                if pred[batch][token] == gt[batch][token]:
                    matches = matches + 1
            accuracies.append(float(matches) / float(count))
        return np.mean(accuracies)

    def save(self):
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                      Save model!                      ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        self.writeParam('current_epoch', str(self.current_epoch))
        saver = tf.train.Saver()
        saver.save(self.session, self.save_path)
        self.printGlobalVariables()
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                     Model saved!                      ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')

    def saveSeeds(self):
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                      Save seeds!                      ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        saver = tf.train.Saver()
        saver.save(self.session, self.seed_path)
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                     Seeds saved!                      ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')

    def writeParam(self, path, value):
        params_path = self.model_dir + '/params'
        if not os.path.exists(params_path):
            os.makedirs(params_path)
        write_path = params_path + '/' + path
        shutil.rmtree(write_path, ignore_errors=True)
        writer = open(write_path, 'w')
        writer.write(value)
        writer.close()

    def readParam(self, path):
        read_path = self.model_dir + '/params/' + path
        if not os.path.exists(read_path):
            print(read_path + ' does not exist!')
            quit()
        reader = open(read_path, 'r')
        value = reader.read()
        reader.close()
        return value

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

    def printVariablesInScope(self, scope):
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            x = self.session.run(v)
            print(v.name + ' : ' + str(v.get_shape()) + ' : ' + str(np.mean(x)) + ' : ' \
                + str(np.var(x)))

    def countVariables(self):
        return np.sum([np.prod(v.get_shape().as_list()) \
            for v in tf.trainable_variables()])

    def countVariablesInScope(self, scope):
        return np.sum([np.prod(v.get_shape().as_list()) \
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)])

    def __init2__(self,  model_dir, num_classes=None, max_num_tokens=None, \
            vocabulary=[], capacity=None, learning_rate=None, \
            train_mode=None, loaded=False, session=None, num_features=None, nr_epochs=1000):
        old = False
        # intialise class variables
        #
        if train_mode == None and not loaded:
            print('either trainmode must be given or model must be loaded!')
            quit()
        #
        if old:
            self.model_dir = model_dir
            self.writeParam('model_dir',model_dir)
            self.train_mode = train_mode
            self.writeParam('train_mode',train_mode)
        else:
            if train_mode == None:
                self.model_dir = model_dir
                self.train_mode = self.readParam('train_mode')
            elif loaded:
                print('trainmode can not be changed!')
                quit()
            else:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                model_dir = model_dir + '/' + train_mode + '_' + \
                    str(datetime.datetime.now())
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                self.model_dir = model_dir
                self.writeParam('model_dir',model_dir)
                self.train_mode = train_mode
                self.writeParam('train_mode',train_mode)

        #code.interact(local=dict(globals(), **locals()))
        #
        if not loaded:
            # create the model directory
            
            self.model_dir = model_dir
        else:
            self.model_dir = model_dir
            
        
        #
        if num_classes == None:
            if old:
                params = np.load(model_dir + '/params.npz')
                num_classes = np.asscalar(params['num_classes'])
                self.writeParam('num_classes',str(num_classes))
            self.num_classes = int(self.readParam('num_classes'))
        elif loaded:
            print('number of classes can not be changed!')
            quit()
        else:
            self.num_classes = num_classes
            self.writeParam('num_classes',str(num_classes))
        
        #
        if vocabulary == []:
            if old:
                params = np.load(model_dir + '/params.npz')
                vocabulary = np.asarray(params['vocabulary'])
                vocabs = ''
                for vocab in vocabulary:
                    vocabs = vocabs + vocab + '\n'
                self.writeParam('vocabulary',str(vocabs[:-1]))
            self.vocabulary = np.array(self.readParam('vocabulary').split('\n'))
        elif loaded:
            print('vocabulary can not be changed!')
            quit()
        else:
            self.vocabulary = np.array(vocabulary.split('\n'))
            self.writeParam('vocabulary',str(vocabulary))
        
        #
        if capacity == None:
            self.capacity = int(self.readParam('capacity'))
        else:
            self.capacity = capacity
            self.writeParam('capacity',str(capacity))
        
        #
        #code.interact(local=dict(globals(), **locals()))
        if max_num_tokens == None:
            self.max_num_tokens = int(self.readParam('max_num_tokens'))
        else:
            self.max_num_tokens = max_num_tokens
            self.writeParam('max_num_tokens',str(max_num_tokens))
        
        #
        if learning_rate == None:
            self.learning_rate = float(self.readParam('learning_rate'))
        else:
            self.learning_rate = float(learning_rate)
            self.writeParam('learning_rate',str(learning_rate))

        #
        if loaded:
            if old:
                params = np.load(model_dir + '/params.npz')
                current_epoch = np.asscalar(params['current_epoch'])
                self.writeParam('current_epoch',str(current_epoch))
            self.current_epoch = int(self.readParam('current_epoch'))
        else:
            self.current_epoch = 0
            self.writeParam('current_epoch',str(self.current_epoch))
        
        #
        self.nr_epochs = nr_epochs

        #
        if num_features == None:
            if old:
                if train_mode[:9] == 'alexnetFe':
                    self.num_features = 256
                else:
                    self.num_features = 512
                self.writeParam('num_features',str(self.num_features))
            self.num_features = int(self.readParam('num_features'))
        elif loaded:
            print('number of features can not be changed!')
            quit()
        else:
            self.num_features = num_features
            self.writeParam('num_features',str(num_features))

        # create necessary paths
        self.logs = self.model_dir + '/logs'
        if not os.path.exists(self.logs):
            os.makedirs(self.logs)
        self.logfile_dir = self.model_dir + '/logfile'
        if not os.path.exists(self.logfile_dir):
            os.makedirs(self.logfile_dir)
        if session == None:
            self.session = tf.Session()
        else:
            self.session = session
        # create the network graph depending on train modes
        self.prediction = tf.constant(-1)
        self.classes_gold = None
        self.save_freq = 1000
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, [])
        # define the feature extractor
        ts = self.train_mode.split('_')
        if ts[3] == 'e2e':
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
            if ts[0] == 'wysiwygFe':
                self.features = WYSIWYGFeatureExtractor.createGraph(self, self.input)
            elif ts[0] == 'alexnetFe':
                self.features = AlexnetFeatureExtractor.createGraph(self, self.input)
            elif ts[0] == 'vggFe':
                self.features = VGGFeatureExtractor.createGraph(self, self.input)
            elif ts[0] == 'resnetFe':
                self.features = ResnetFeatureExtractor.createGraph(self, self.input)
            elif ts[0] == 'densenetFe':
                self.features =DensenetFeatureExtractor.createGraph(self, self.input)
            else:
                print('e2e must specify an feature extractor!')
                quit()
        else:
            self.input = tf.placeholder(dtype=tf.float32, shape= [None, None, None, \
                self.num_features])
            self.features = self.input
        #
        if ts[1] == '' and ts[2] == '':
            print('classification!')
            #self.classes_gold = tf.placeholder(dtype=tf.float32, shape=[None,num_classes])
            self.groundtruth = tf.placeholder(dtype=tf.float32, shape=[None,num_classes])
            self.classes_gold = self.groundtruth
            if ts[0] == 'wysiwygFe':
                self.prediction = WYSIWYGClassifier.createGraph(self, self.features)
            elif ts[0] == 'alexnetFe':
                self.prediction = AlexnetClassifier.createGraph(self, self.features)
            elif ts[0] == 'vggFe':
                self.prediction = VGGClassifier.createGraph(self, self.features)
            elif ts[0] == 'resnetFe':
                self.prediction = ResnetClassifier.createGraph(self, \
                    self.features)
            elif ts[0] == 'densenetFe':
                self.prediction = DensenetClassifier.createGraph(self, self.features)
            else:
                print('e2e must specify an feature extractor!')
                quit()
            self.useClassesLoss()
        elif ts[1] != '' and ts[2] != '':
            print('captioning!')
            #self.label_gold = tf.placeholder(dtype=tf.int32, shape=[None,max_num_tokens])
            self.groundtruth = tf.placeholder(dtype=tf.int32, shape=[None,max_num_tokens])
            self.label_gold = self.groundtruth
            # define the encoder
            if ts[1] == 'simpleEnc':
                outputs, state = SimpleEncoder.createGraph(self, self.features)
            elif ts[1] == 'monorowEnc':
                outputs, state = MonorowEncoder.createGraph(self, self.features)
            elif ts[1] == 'birowEnc':
                outputs, state = BirowEncoder.createGraph(self, self.features)
            elif ts[1] == 'monocolEnc':
                outputs, state = MonocolEncoder.createGraph(self, self.features)
            elif ts[1] == 'bicolEnc':
                outputs, state = BicolEncoder.createGraph(self, self.features)
            elif ts[1] == 'quadroEnc':
                outputs, state = QuadroEncoder.createGraph(self, self.features)
            else:
                print(ts[1] + ' is no valid encoder type!')
                quit()
            # define the decoder
            if ts[2] == 'simpleDec':
                self.prediction = SimpleDecoder(self).createGraph(outputs, state)
            elif ts[2] == 'simplegruDec':
                self.prediction = SimplegruDecoder(self).createGraph(outputs, state)
            elif ts[2] == 'bahdanauDec':
                self.prediction = BahdanauDecoder(self).createGraph(outputs, state)
            elif ts[2] == 'luongDec':
                self.prediction = LuongDecoder(self).createGraph(outputs, state)
            else:
                print(ts[2] + ' is no valid decoder type!')
                quit()
            self.useLabelLoss()
        else:
            print('encoder and decoder must be used together!')
            quit()
        # different attention modules!!!
        assert self.input != None
        assert self.loss != None
        self.optimizer ='sgd'
        self.createOptimizer()
        # create the save path
        self.save_path = self.model_dir + '/weights.ckpt'
        self.seed_path = self.model_dir + '/seeds.ckpt'
        # initialises all variables
        if not loaded:
            init = tf.global_variables_initializer()
            self.session.run(init)
            for v in tf.trainable_variables():
                #code.interact(local=dict(globals(), **locals()))
                seed = np.random.standard_normal(v.shape.as_list()) * 0.05
                v = v.assign(seed)
                x = self.session.run(v)
                #print(seed.shape)
            self.save()
            self.saveSeeds()
        #code.interact(local=dict(globals(), **locals()))
        tf.summary.histogram('predictionHisto', self.predictionDistribution)
        tf.summary.scalar('predictionAvg', tf.reduce_mean(self.predictionDistribution))
        tf.summary.tensor_summary('predictionTensor', self.predictionDistribution)
        self.summaries = tf.summary.merge_all()
        self.board_path = self.model_dir + '/tensorboard'
        self.writer = tf.summary.FileWriter(self.board_path, graph=tf.get_default_graph())
        self.step = 0
        self.predictionDistributionsDone = 0
        param_count_path = self.model_dir + '/param_count.txt'
        shutil.rmtree(param_count_path, ignore_errors=True)
        param_count_writer = open(param_count_path, 'w')
        param_count_writer.write(str(self.countVariables()))
        param_count_writer.close()

    # indicates to use the momentum optimizer
    def useGradientDescentOptimizer(self, decayed_learning_rate=None):
        if decayed_learning_rate != None:
            self.learning_rate = decayed_learning_rate
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
        self.optimzer_name = "MyAdamOptimizer"
        with tf.variable_scope(self.optimzer_name, reuse=None):
            optimizer = tf.train.AdamOptimizer()
            #code.interact(local=dict(globals(), **locals()))
            self.train_op = slim.learning.create_train_op(self.loss, optimizer, \
                summarize_gradients=True)

    # indicates to use the momentum optimizer
    def useAdadeltaOptimizer(self):
        self.optimizer_name = "MyAdadeltaOptimizer"
        with tf.variable_scope(self.optimizer_name, reuse=None):
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            #code.interact(local=dict(globals(), **locals()))
            self.train_op = slim.learning.create_train_op(self.loss, optimizer, \
                summarize_gradients=True)

    # indicates to fit the predicted label to the gold label as objective
    def useLabelLoss(self):
        try:
            with tf.variable_scope("labelLoss", reuse=None):
                batchsize = tf.shape(self.prediction)
                self.predictionDistribution = tf.nn.softmax(self.prediction)
                self.prediction = tf.argmax(self.predictionDistribution, axis=2)
                diff = self.max_num_tokens - tf.shape(self.prediction)[1]
                self.prediction = tf.pad(self.prediction, [[0,0],[0,diff]])
                self.used_loss = 'label'
                eos_gt = tf.argmax(self.groundtruth, 1)
                eos_pred = tf.argmax(self.prediction, 1)
                max_seq_length = tf.maximum(eos_gt, eos_pred)
                weights = tf.sequence_mask(max_seq_length, self.max_num_tokens)
                weights = tf.cast(weights, tf.float32)
                self.loss = tf.contrib.seq2seq.sequence_loss(self.prediction,\
                    self.groundtruth, weights)
        except Exception:
            print('exception in label loss')
            print(sys.exc_info())
            code.interact(local=dict(globals(), **locals()))

    # indicates to fit the predicted classes to the gold classes as objective
    def useClassesLoss(self):
        with tf.variable_scope("classesLoss", reuse=None):
            #self.groundtruth = self.classes_gold            
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.groundtruth, \
                logits=self.prediction)
            self.loss = tf.reduce_mean(loss)
            self.prediction = tf.sigmoid(self.prediction)
            self.predictionDistribution = self.prediction
            self.prediction = tf.round(self.predictionDistribution)
            self.used_loss = 'classes'

def load(model_dir, train_mode=None, max_num_tokens=None, learning_rate=None, capacity=None, \
        num_classes=None, combine_models=False, fe_dir=None, \
        enc_dec_dir=None):
    tf.reset_default_graph()
    session = tf.Session()    
    if not combine_models:
        old = False
        if old:
            params = np.load(model_dir + '/params.npz')
            capacity = np.asscalar(params['capacity'])
            learning_rate = np.asscalar(params['learning_rate'])
        else:
            train_mode = None
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                 Try to restore model!                 ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        model = Model(model_dir, max_num_tokens=max_num_tokens, \
            capacity=capacity, learning_rate=learning_rate, \
            train_mode=train_mode, loaded=True, \
            session=session)    
        print('load variables!')
        saver = tf.train.Saver()
        saver.restore(session, model_dir + '/weights.ckpt')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                    Model restored!                    ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
    else:
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                Try to combine models!                 ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        #code.interact(local=dict(globals(), **locals()))
        variables = {}
        fe = load(fe_dir)
        #metagraph = tf.train.import_meta_graph(fe_dir + '/weights.ckpt.meta')
        #metagraph.restore(session, fe_dir + '/weights.ckpt')
        for v in tf.global_variables():
            value = fe.session.run(v)
            #scode.interact(local=dict(globals(), **locals()))
            variables.update({str(v.name): value})
        tf.reset_default_graph()
        encoder_decoder = load(enc_dec_dir)
        #metagraph = tf.train.import_meta_graph(enc_dec_dir + '/weights.ckpt.meta')
        #metagraph.restore(session, enc_dec_dir + '/weights.ckpt')
        for v in tf.global_variables():
            value = encoder_decoder.session.run(v)
            variables.update({str(v.name): value})
        tf.reset_default_graph()
        model = Model(model_dir, fe.num_classes, fe.max_num_tokens, \
            fe.readParam('vocabulary'), capacity= capacity, train_mode=train_mode, \
            loaded=False, learning_rate=learning_rate, num_features=fe.num_features)
        for v in tf.trainable_variables():
            if v.name in variables.keys():
                v = v.assign(variables[v.name])
                x = model.session.run(v)
        model.printGlobalVariables()
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                   Models combined!                    ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
    return model