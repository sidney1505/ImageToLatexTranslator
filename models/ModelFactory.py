import code, os, shutil, tflearn, datetime, sys
import tensorflow as tf
import numpy as np
# import feature extractors
from FeatureExtractors.WYSIWYGFeatureExtractor import WYSIWYGFeatureExtractor
from FeatureExtractors.AlexnetFeatureExtractor import AlexnetFeatureExtractor
from FeatureExtractors.VGGFeatureExtractor import VGGFeatureExtractor
from FeatureExtractors.VGGFinegrainedFeatureExtractor import VGGFinegrainedFeatureExtractor
from FeatureExtractors.ResnetFeatureExtractor import ResnetFeatureExtractor
from FeatureExtractors.DensenetFeatureExtractor import DensenetFeatureExtractor
from FeatureExtractors.VGGLevelFeatureExtractor import VGGLevelFeatureExtractor
# import classifiers
from Classifiers.SimpleClassifier import SimpleClassifier
# import encoders
from Encoders.SimpleEncoder import SimpleEncoder
from Encoders.MonorowEncoder import MonorowEncoder
from Encoders.BirowEncoder import BirowEncoder
from Encoders.MonocolEncoder import MonocolEncoder
from Encoders.BicolEncoder import BicolEncoder
from Encoders.RowcolEncoder import RowcolEncoder
from Encoders.QuadroEncoder import QuadroEncoder
from Encoders.StackedQuadroEncoder import StackedQuadroEncoder
from Encoders.LevelEncoder import LevelEncoder
# import decoders
from Decoders.SimpleDecoder import SimpleDecoder
from Decoders.SimplegruDecoder import SimplegruDecoder
from Decoders.BahdanauDecoder import BahdanauDecoder
from Decoders.BahdanauMonotonicDecoder import BahdanauMonotonicDecoder
from Decoders.StackedBahdanauDecoder import StackedBahdanauDecoder
from Decoders.StackedEncBahdanauDecoder import StackedEncBahdanauDecoder
from Decoders.LuongDecoder import LuongDecoder
from Decoders.StackedEncLuongDecoder import StackedEncLuongDecoder
from Decoders.LuongMonotonicDecoder import LuongMonotonicDecoder

class Model:
    def __init__(self, model_dir, feature_extractor='', encoder='',\
            decoder='', encoder_size=0, decoder_size=0, optimizer='', \
            learning_rate=0, vocabulary='', max_num_tokens=None, \
            capacity=-1, loaded=False, only_inference=False):
        tf.reset_default_graph()
        self.session = tf.Session()
        # temporory solution
        self.num_features = 512
        #
        if loaded:
            # load parameters from given directory
            self.model_dir = model_dir
            #if model_dir.split('/')[-1][0] == '_':
            #    code.interact(local=dict(globals(), **locals()))
            # unchangable parameters
            self.feature_extractor = self.readParam('feature_extractor')
            self.encoder = self.readParam('encoder')
            self.decoder = self.readParam('decoder')
            self.encoder_size = int(self.readParam('encoder_size'))
            self.decoder_size = int(self.readParam('decoder_size'))
            self.vocabulary = np.array(self.readParam('vocabulary').split('\n'))
            # changable parameters
            if optimizer == '':
                self.optimizer = self.readParam('optimizer')
            else:
                self.optimizer = optimizer
                self.writeParam('optimizer',str(optimizer))
            if learning_rate == 0:
                self.learning_rate = float(self.readParam('learning_rate'))
            else:
                self.learning_rate = learning_rate
                self.writeParam('learning_rate',str(learning_rate))
            if max_num_tokens == None:
                self.max_num_tokens = int(self.readParam('max_num_tokens'))
            else:
                self.max_num_tokens = max_num_tokens
                self.writeParam('max_num_tokens',str(max_num_tokens))
            if capacity <= 0:
                self.capacity = int(self.readParam('capacity'))
            else:
                self.capacity = capacity
                self.writeParam('capacity',str(capacity))
            # implicit parameters
            self.current_epoch = int(self.readParam('current_epoch'))
            self.current_step = int(self.readParam('current_step'))
            self.current_millis = float(self.readParam('current_millis'))
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
            #
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
            self.writeParam('current_epoch',str(self.current_epoch))
            self.current_step = 0
            self.writeParam('current_step',str(self.current_step))
            self.current_millis = 0
            self.writeParam('current_millis',str(self.current_millis))
        self.num_classes = len(self.vocabulary)
        self.only_inference = only_inference
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
            elif self.feature_extractor == 'vggfFe':
                self.features = VGGFinegrainedFeatureExtractor(self).createGraph()
            elif self.feature_extractor == 'resnetFe':
                self.features = ResnetFeatureExtractor(self).createGraph()
            elif self.feature_extractor == 'densenetFe':
                self.features =DensenetFeatureExtractor(self).createGraph()
            elif self.feature_extractor == 'levelFe':
                self.features =VGGLevelFeatureExtractor(self).createGraph()
            else:
                print(self.feature_extractor + ' is no valid feature extractor!')
                quit()
        else:
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, \
                self.num_features])
            self.features = self.input
        if self.encoder != '' and self.decoder != '' and self.encoder_size != 0 and \
                self.decoder_size != 0:
            if not only_inference:
                #print('in model factory constructor')
                #code.interact(local=dict(globals(), **locals()))
                self.groundtruth = tf.placeholder(dtype=tf.int32, shape=[None, \
                    self.max_num_tokens])
            # define the encoder
            if self.encoder == 'monorowEnc':
                #MonorowEncoder(self).createGraph()
                self.monoenc = MonorowEncoder(self)
                self.monoenc.createGraph()
            elif self.encoder == 'simpleEnc':
                SimpleEncoder(self).createGraph()
            elif self.encoder == 'birowEnc':
                BirowEncoder(self).createGraph()
            elif self.encoder == 'monocolEnc':
                #MonocolEncoder(self).createGraph()
                self.monoenc = MonocolEncoder(self)
                self.monoenc.createGraph()
            elif self.encoder == 'bicolEnc':
                BicolEncoder(self).createGraph()
            elif self.encoder == 'rowcolEnc':
                RowcolEncoder(self).createGraph()
            elif self.encoder == 'quadroEnc':
                QuadroEncoder(self).createGraph()
            elif self.encoder == 'stackedquadroEnc':
                StackedQuadroEncoder(self).createGraph()
            elif self.encoder == 'levelEnc':
                LevelEncoder(self).createGraph()
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
            elif self.decoder == 'monobahdanauDec':
                BahdanauMonotonicDecoder(self).createGraph()
            elif self.decoder == 'luongDec':
                LuongDecoder(self).createGraph()
            elif self.decoder == 'monoluongDec':
                LuongMonotonicDecoder(self).createGraph()
            elif self.decoder == 'stackedbahdanauDec':
                StackedBahdanauDecoder(self).createGraph()
            elif self.decoder == 'stackedencbahdanauDec':
                StackedEncBahdanauDecoder(self).createGraph()
            elif self.decoder == 'stackedencluongDec':
                StackedEncLuongDecoder(self).createGraph()
            else:
                print(self.decoder + ' is no valid decoder type!')
                quit()
            self.used_loss = 'label'
            if not only_inference:
                self.__useLabelLoss()
        elif self.encoder != '' or self.decoder != '' or self.encoder_size != 0 or \
                self.decoder_size != 0:
            print('encoder and decoder must be used together!')
            quit()
        else:
            SimpleClassifier(self).createGraph()
            self.used_loss = 'classes'
            if not only_inference:
                self.groundtruth = tf.placeholder(dtype=tf.int32, shape=[None, \
                    self.num_classes])
                self.__useClassesLoss()
        # creates the optimizer
        # create the save path
        self.save_path = self.model_dir + '/weights.ckpt'
        self.best_path = self.model_dir + '/best.ckpt'
        self.seed_path = self.model_dir + '/seeds.ckpt'
        # initialises all variables
        if not loaded:
            if not only_inference:
                self.createOptimizer()
            init = tf.global_variables_initializer()
            self.session.run(init)
            for v in tf.trainable_variables():
                #code.interact(local=dict(globals(), **locals()))
                seed = np.random.standard_normal(v.shape.as_list()) * 0.05
                v = v.assign(seed)
                x = self.session.run(v)
                #print(seed.shape)
            self.save()
            self.__saveSeeds()
        # code.interact(local=dict(globals(), **locals()))
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
        self.createOptimizer(reuse=True)

    def getTrainMode(self):
        return self.feature_extractor + '_' + self.encoder + '_' + self.decoder \
            + '_' + str(self.encoder_size) + '_' + str(self.decoder_size) + '_' + \
            self.optimizer

    def trainStep(self, inp, groundtruth):
        feed_dict={self.input: inp, self.groundtruth: groundtruth, self.is_training:True, \
            self.keep_prob:0.5}
        # code.interact(local=dict(globals(), **locals()))
        #code.interact(local=dict(globals(), **locals()))

        _,self.current_train_loss, self.current_infer_loss, self.current_train_prediction, \
            self.current_infer_prediction = self.session.run([\
            self.update_step, self.train_loss, self.infer_loss, self.train_prediction, \
            self.infer_prediction], feed_dict=feed_dict)
        if self.current_step % 1000 == 0 and self.current_step != 0:
            print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
            print('global step: ' + str(self.current_step))
            print('Current Weights:')
            self.printTrainableVariables()
            print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
            self.save()
        '''if self.used_loss == 'label':
            self.current_train_prediction = self.__argmaxs(self.current_train_prediction)
            self.current_infer_prediction = self.__argmaxs(self.current_infer_prediction)
            self.current_infer_prediction = \
                self.__fillWithEndTokens(self.current_infer_prediction)
            self.current_infer_prediction = self.current_infer_prediction.astype(int)'''
        self.current_train_accuracy = self.calculateAccuracy( \
            self.current_train_prediction, groundtruth)
        self.current_infer_accuracy = self.calculateAccuracy( \
            self.current_infer_prediction, groundtruth)
        self.current_step  = self.current_step + 1

    def valStep(self, inp, groundtruth):
        feed_dict={self.input: inp, self.groundtruth:groundtruth,\
            self.is_training:False, self.keep_prob:1.0}
        self.current_train_loss, self.current_infer_loss,self.current_train_prediction,\
            self.current_infer_prediction = self.session.run([self.train_loss, \
            self.infer_loss, self.train_prediction, self.infer_prediction], \
            feed_dict=feed_dict)
        self.current_train_accuracy = self.calculateAccuracy( \
            self.current_train_prediction, groundtruth)
        self.current_infer_accuracy = self.calculateAccuracy( \
            self.current_infer_prediction, groundtruth)

    def testStep(self, inp):
        feed_dict={self.input: inp, self.is_training:False, self.keep_prob:1.0}
        self.current_infer_prediction = self.session.run(self.infer_prediction, \
            feed_dict=feed_dict)
        #top_k = self.session.run(self.top_k, feed_dict=feed_dict)
        #print(top_k.shape)
        #code.interact(local=dict(globals(), **locals()))

    def predict(self, wanted, inp):
        feed_dict={self.input: inp, self.is_training:False, self.keep_prob:1.0}
        prediction = self.session.run(wanted, feed_dict=feed_dict)
        return prediction

    def restoreLastCheckpoint(self):
        saver = tf.train.Saver()
        # self.createOptimizer()
        saver.restore(self.session, self.save_path)

    def restoreBestCheckpoint(self):
        saver = tf.train.Saver()
        # self.createOptimizer()
        saver.restore(self.session, self.best_path)

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

    def saveBest(self):
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                 Save best model!                      ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        saver = tf.train.Saver()
        saver.save(self.session, self.best_path)
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                 Best Model saved!                     ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')

    def __saveSeeds(self):
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
        params_path = self.model_dir + '/params/' + '/'.join(path.split('/')[:-1])
        if not os.path.exists(params_path):
            os.makedirs(params_path)
        write_path = params_path + '/' + path.split('/')[-1]
        shutil.rmtree(write_path, ignore_errors=True)
        writer = open(write_path, 'w')
        writer.write(str(value))
        writer.close()

    def writeParamInList(self, path, value):
        params_path = self.model_dir + '/params/' + '/'.join(path.split('/')[:-1])
        if not os.path.exists(params_path):
            os.makedirs(params_path)
        write_path = params_path + '/' + path.split('/')[-1]
        writer = open(write_path, 'a')
        writer.write(str(value) + '\n')
        writer.close()

    def readParamList(self, path):
        read_path = self.model_dir + '/params/' + path
        if not os.path.exists(read_path):
            print(read_path + ' does not exist!')
            return []
        reader = open(read_path, 'r')
        value = reader.read().split('\n')[:-1]
        reader.close()
        return value

    def readParam(self, path):
        read_path = self.model_dir + '/params/' + path
        if not os.path.exists(read_path):
            raise Exception(read_path + ' does not exist!')
        reader = open(read_path, 'r')
        value = reader.read()
        reader.close()
        return value

    def printTrainableVariables(self):
        for v in tf.trainable_variables():
            x = self.session.run(v)
            print(v.name + ' : ' + str(v.get_shape()) + ' : ' + str(np.mean(x)) + ' : ' \
                + str(np.var(x)))

    def printTrainableVariablesLight(self):
        for v in tf.trainable_variables():
            print(v.name + ' : ' + str(v.get_shape()))

    def printTrainableVariablesInScope(self, scope):
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            if v in tf.trainable_variables():
                x = self.session.run(v)
                print(v.name + ' : ' + str(v.get_shape()) + ' : ' + str(np.mean(x)) + ' : ' \
                    + str(np.var(x)))

    def printGlobalVariables(self):
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
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

    def createOptimizer(self, reuse=None):
        with tf.variable_scope("optimizer", reuse=reuse):
            if self.optimizer == 'momentum':
                #with tf.variable_scope("MyMomentumOptimizer", reuse=reuse):
                optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9)                
            elif self.optimizer == 'sgd':
                #with tf.variable_scope("MySGDOptimizer", reuse=reuse):
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == 'adam':
                #with tf.variable_scope("MyAdamOptimizer", reuse=reuse):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                #with tf.variable_scope("MyAdadeltaOptimizer", reuse=reuse):
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            params = tf.trainable_variables()
            gradients = tf.gradients(self.train_loss, params)
            # gradients = tf.gradients(self.infer_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 2.0) # in [1,5]
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
            l = []
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer'):
                #print(v.name)
                l.append(v)
            init = tf.variables_initializer(l)
            self.session.run(init)
            # code.interact(local=dict(globals(), **locals()))

    # indicates to fit the predicted label to the gold label as objective
    def __useLabelLoss(self):
        with tf.variable_scope("labelLoss", reuse=None):
            gt = tf.transpose(self.groundtruth)
            gt_train = gt[:tf.shape(self.train_energy)[1]]
            gt_train = tf.transpose(gt_train)
            
            #diff = self.max_num_tokens - tf.shape(self.train_pred)[1]
            #self.prediction_greedy = tf.pad(self.train_pred, [[0,0],[0,diff]])
            eos_gt = tf.argmax(gt_train, 1)
            eos_pred = tf.argmax(self.train_prediction, 1)
            max_seq_length = tf.maximum(eos_gt, eos_pred)
            weights = tf.sequence_mask(max_seq_length, tf.shape(gt_train)[-1])
            weights = tf.cast(weights, tf.float32)
            #code.interact(local=dict(globals(), **locals()))
            self.train_loss = tf.contrib.seq2seq.sequence_loss(self.train_energy,\
                gt_train, weights)
            #
            gt_infer = gt[:tf.shape(self.infer_energy)[1]]
            gt_infer = tf.transpose(gt_infer)
            #diff = self.max_num_tokens - tf.shape(self.infer_pred)[1]
            #self.prediction_greedy = tf.pad(self.infer_pred, [[0,0],[0,diff]])
            eos_gt = tf.argmax(gt_infer, 1)
            eos_pred = tf.argmax(self.infer_prediction, 1)
            max_seq_length = tf.maximum(eos_gt, eos_pred)
            weights = tf.sequence_mask(max_seq_length, tf.shape(gt_infer)[-1])
            weights = tf.cast(weights, tf.float32)
            self.infer_loss = tf.contrib.seq2seq.sequence_loss(self.infer_energy,\
                gt_infer, weights)
        #except Exception:
        #    print('exception in label loss')
        #    print(sys.exc_info())
        #    code.interact(local=dict(globals(), **locals()))

    # temporary solution
    def setMaxNumTokens(self, new_max_num_tokens):
        if self.max_num_tokens != new_max_num_tokens:
            self.max_num_tokens = new_max_num_tokens
            self.groundtruth = tf.placeholder(dtype=tf.int32, shape=[None, \
                self.max_num_tokens])
            self.__useLabelLoss()

    # indicates to fit the predicted classes to the gold classes as objective
    def __useClassesLoss(self):
        with tf.variable_scope("classesLoss", reuse=None):
            #code.interact(local=dict(globals(), **locals()))
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float( \
                self.groundtruth), logits=self.prediction)
            self.train_loss = tf.reduce_mean(loss)
            self.infer_loss = self.train_loss
            self.prediction = tf.sigmoid(self.prediction)
            self.train_prediction = tf.round(self.prediction)
            self.infer_prediction = self.train_prediction

    def __fillWithEndTokens(self, prediction):
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

    def calculateAccuracy(self, pred, gt):
        accuracies = []
        for batch in range(gt.shape[0]):
            #print('|||||||||||||||||||||||||||||||||||||')
            matches = 0
            count = 0
            for token in range(gt.shape[1]):
                if token >= pred.shape[1]:
                    #print('break at token '+ str(token))
                    #print('shape: '+ str(pred.shape))
                    #print(str(matches) + '/' + str(count))
                    if gt[batch][token] == self.num_classes - 1:
                        break
                    else:
                        count = count + 1
                        continue
                if pred[batch][token] == self.num_classes - 1 and \
                        gt[batch][token] == self.num_classes - 1:
                    #print('double break at token '+ str(token))
                    #print(str(matches) + '/' + str(count))
                    break
                count = count + 1
                if pred[batch][token] == gt[batch][token]:
                    matches = matches + 1
                    #print('Match' + str(pred[batch][token]) + ' : ' +  str(gt[batch][token]))
                #else:
                    #print('Miss ' + str(pred[batch][token]) + ' : ' +  str(gt[batch][token]))
                # code.interact(local=dict(globals(), **locals()))
            if count == 0:
                accuracies.append(0.0)
            else:
                accuracies.append(float(matches) / float(count))
        return np.mean(accuracies)