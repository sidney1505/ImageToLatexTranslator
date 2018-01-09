import sys, os, argparse, logging, time
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['xtick.labelsize'] = 7
import matplotlib.pyplot as plt
import numpy as np
import distance
import code
import shutil
import datetime
# from model.model import Model
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
# own packages
import myutils.render_output as my_renderer
from models.ModelFactory import Model
import models.ModelFactory
from Evaluator import Evaluator

class Trainer:
    def __init__(self, model_dir, dataset_dir, tmp_dir, capacity):
        self.model = None
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        vocabulary_path = self.dataset_dir + '/vocabulary.txt'
        self.vocabulary = open(vocabulary_path).read() + '\n' + 'END'
        assert os.path.exists(self.dataset_dir), \
            "Dataset directory doesn't exists!"
        self.tmp_dir = tmp_dir
        self.capacity = capacity / 4
        # standart values for the hyperparameters
        self.mode = 'stepbystep'
        self.feature_extractor = 'vggfFe'
        self.encoder = 'quadroEnc'
        self.decoder = 'bahdanauDec'
        self.encoder_size = 2048
        self.decoder_size = 512
        self.optimizer = 'momentum'
        self.min_epochs = 6
        self.max_epochs = 50
        self.initial_learning_rate = 0.1
        self.lr_decay = 0.1
        # indicates whether and which preprocessed batches should be used
        self.preprocessing = ''
        self.best = True

    def setModelParameters(self, mode, feature_extractor, encoder, decoder, encoder_size, \
            decoder_size, optimizer):
        self.mode = mode
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.optimizer = optimizer

    def trainModel(self, proceed=True):
        if self.mode == 'e2e':
            self.model = Model(self.model_dir, self.feature_extractor, \
                self.encoder, self.decoder, self.encoder_size, \
                self.decoder_size, self.optimizer, self.initial_learning_rate, \
                self.vocabulary, self.__getMaxNumTokens(self.__getBatchDir('train')), \
                self.capacity)
            self.train()
        elif self.mode == 'stepbystep':
            fe_dir = self.__findBestModel(self.feature_extractor, optimizer=self.optimizer)
            if fe_dir == None or not proceed:
                self.model = Model(self.model_dir, self.feature_extractor, optimizer= \
                    self.optimizer, vocabulary=self.vocabulary, capacity=self.capacity, \
                    max_num_tokens =self.__getMaxNumTokens(self.__getBatchDir('train')), \
                    learning_rate=self.initial_learning_rate)
                self.train()
                self.testModel()
                fe_dir = self.model.model_dir
            else:
                self.loadModel(fe_dir)
            preprocessing = self.model.feature_extractor + 'Preprocessed_'
            basedir = self.dataset_dir + "/" + preprocessing
            if not os.path.exists(basedir + 'train_batches') or \
                    not os.path.exists(basedir + 'val_batches') or \
                    not os.path.exists(basedir + 'test_batches') or \
                    not proceed:
                # code.interact(local=dict(globals(), **locals()))
                self.processData(preprocessing)
            self.preprocessing = preprocessing
            ed_dir = self.__findBestModel(feature_extractor=self.preprocessing[:-1], \
                encoder=self.encoder, decoder=self.decoder, \
                encoder_size=self.encoder_size, decoder_size=self.decoder_size, \
                optimizer=self.optimizer)
            if ed_dir == None or not proceed:
                self.model = Model(self.model_dir, encoder=self.encoder, \
                    decoder=self.decoder, encoder_size=self.encoder_size, \
                    decoder_size=self.decoder_size, optimizer=self.optimizer, \
                    vocabulary=self.vocabulary, capacity=self.capacity, \
                    max_num_tokens =self.__getMaxNumTokens(self.__getBatchDir('train')), \
                    learning_rate=self.initial_learning_rate)
                self.train()
                self.testModel()
                ed_dir = self.model.model_dir
            self.preprocessing = ''
            self.combineModels(fe_dir, ed_dir)
            # code.interact(local=dict(globals(), **locals()))
            self.train()
        self.testModel()
        self.evaluateModel()

    '''def evaluateModel(self):
        test_result_path = self.model.model_dir + '/params/test_results/epoch' + \
            str(self.model.current_epoch)
        test_image_path = self.model.model_dir + 'rendered_test_images'
        my_renderer.render_output(test_result_path, test_image_path)
        e_test = Evaluator(self.model.model_dir)
        e_test.evaluate()
        val_result_path = self.model.model_dir + '/params/val_results/epoch' + \
            str(self.model.current_epoch - 1)
        val_image_path = self.model.model_dir + 'rendered_val_images'
        my_renderer.render_output(val_result_path, val_image_path)
        e_test = Evaluator(self.model.model_dir)
        e_test.evaluate()'''

    def processData(self, preprocessing):
        for phase in ['train','val','test']:
            path = self.dataset_dir + "/" + preprocessing + '_' + \
                    phase + '_batches'
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path)
            batch_it = 0
            batch_images, minibatchsize, batch_it, groundtruth, new_iteration, \
                batch_classes_true, batch_imgnames, batch_labels = self.__loadBatch( \
                phase, batch_it)
            while not new_iteration: # randomise!!!
                print(batch_it)
                minibatch_predictions = []
                for minibatch_it in range(batch_images.shape[0] / minibatchsize + 1):
                    if batch_images.shape[0] == minibatchsize * minibatch_it:
                        break
                    minibatch_images = self.__createMinibatch(batch_images, \
                        minibatch_it, minibatchsize)
                    minibatch_prediction = self.model.predict(self.model.features, \
                        minibatch_images)
                    minibatch_predictions.append(minibatch_prediction)
                batch_prediction = np.concatenate(minibatch_predictions)
                # code.interact(local=dict(globals(), **locals()))
                assert batch_prediction.shape[0] == batch_images.shape[0]
                batch_path = path +  '/batch' + str(batch_it) + '.npz'
                with open(batch_path, 'w') as fout:
                    np.savez(fout, images=batch_prediction, labels=batch_labels, \
                        num_classes=self.model.num_classes, \
                        contained_classes_list=batch_classes_true, \
                        image_names=batch_imgnames)
                print(phase + 'batch' + str(batch_it) + " saved! " + \
                    str(batch_prediction.shape))
                batch_images, minibatchsize, batch_it, groundtruth, new_iteration, \
                    batch_classes_true, batch_imgnames, batch_labels = \
                    self.__loadBatch(phase, batch_it)

    def train(self):
        # the training loop
        self.best = False
        train_last_loss = float('inf')
        val_last_loss = float('inf')
        val_last_infer_loss = float('inf')
        while True:
            begin = time.time()
            train_loss, train_accuracy, infer_loss, infer_accuracy = \
                self.__iterateOneEpoch('train')
            val_loss, val_accuracy, val_infer_loss, val_infer_accuracy = \
                self.__iterateOneEpoch('val')
            '''a = (self.model.current_epoch + 1) / 10.0
            l = 1.0 / a
            train_loss, train_accuracy, infer_loss, infer_accuracy = l,a,l*2,a/2
            val_loss, val_accuracy, val_infer_loss, val_infer_accuracy = l*3,a/3,l*4,a/4'''
            # updates the implicit parameters of the model            
            self.model.current_epoch = self.model.current_epoch + 1
            self.model.writeParam('current_epoch',self.model.current_epoch)
            self.model.writeParam('current_step',self.model.current_step)
            end = time.time()
            self.model.current_millis = self.model.current_millis + (end - begin)
            self.model.writeParam('current_millis',self.model.current_millis)
            self.model.writeParamInList('stats/losses/train_train',train_loss)
            self.model.writeParamInList('stats/losses/train_infer',infer_loss)
            self.model.writeParamInList('stats/losses/val_train',val_loss)
            self.model.writeParamInList('stats/losses/val_infer',val_infer_loss)
            self.model.writeParamInList('stats/accs/train_train',train_accuracy)
            self.model.writeParamInList('stats/accs/train_infer',infer_accuracy)
            self.model.writeParamInList('stats/accs/val_train',val_accuracy)
            self.model.writeParamInList('stats/accs/val_infer',val_infer_accuracy)
            self.saveLossGraph()
            self.saveAccGraph()
            self.saveTrainLossGraph()
            self.saveTrainAccGraph()
            # checks whether model is still learning or overfitting
            if self.model.current_epoch >= self.max_epochs:
                print('model training lasted to long!')
                break
            elif val_last_loss < val_loss and self.model.current_epoch > self.min_epochs:
                print('model starts to overfit!')
                break
            elif val_last_infer_loss < val_infer_loss: # or self.model.current_epoch % 3 == 0:
                self.model.decayLearningRate(self.lr_decay)
            else:
                self.model.saveBest()
            val_last_infer_loss = val_infer_loss
            train_last_loss = train_loss
            val_last_loss = val_loss

    def __iterateOneEpoch(self, phase):
        if self.model.used_loss == 'classes':
            false2false = np.zeros(self.model.num_classes)
            false2true = np.zeros(self.model.num_classes)
            true2true = np.zeros(self.model.num_classes)
            true2false = np.zeros(self.model.num_classes)
        batch_it = 0
        batch_images, minibatchsize, batch_it, groundtruth, new_iteration, \
            batch_classes_true, batch_imgnames, batch_labels = \
            self.__loadBatch(phase, batch_it)
        num_samples = 0
        samples_correct = 0
        tokens_correct = 0
        num_tokens = 0
        lines = ''
        train_losses = []
        train_accuracies = []
        infer_losses = []
        infer_accuracies = []
        while not new_iteration: # randomise!!!
            if batch_labels.shape[-1] != self.model.max_num_tokens:
                self.loadModel(self.model.model_dir, max_num_tokens= \
                    batch_labels.shape[-1], only_inference=self.model.only_inference)
                print('model needed to be reloaded, this will cause massive slowdown!')
                #code.interact(local=dict(globals(), **locals()))
            print(batch_it)
            batch_predictions = []
            for minibatch_it in range(batch_images.shape[0] \
                    / minibatchsize):
                # create the minibatches
                minibatch_images = self.__createMinibatch(batch_images, \
                    minibatch_it, minibatchsize)
                minigroundtruth = self.__createMinibatch(groundtruth, \
                    minibatch_it, minibatchsize)
                minibatch_imgnames = self.__createMinibatch(batch_imgnames, \
                    minibatch_it, minibatchsize)
                # make the prints for the minibatch
                print(phase + 'minibatch(' + str(self.model.current_epoch) + ',' + \
                    str(batch_it - 1) + ',' + str(minibatch_it*minibatchsize)+')[' + \
                    str(self.model.current_step) + '] : '+ str(minibatch_images.shape) + \
                    ' : ' + str(minigroundtruth.shape))
                print(self.preprocessing.split('_')[0] + self.model.model_dir)
                #
                #code.interact(local=dict(globals(), **locals()))
                if phase == 'train':
                    self.model.trainStep(minibatch_images, minigroundtruth)
                elif phase == 'val':
                    self.model.valStep(minibatch_images, minigroundtruth)
                if phase == 'test':
                    self.model.testStep(minibatch_images)
                    self.model.current_train_loss = 0
                    self.model.current_train_accuracy = 0
                    self.model.current_infer_loss = 0
                    self.model.current_infer_accuracy = self.model.calculateAccuracy( \
                        self.model.current_infer_prediction, minigroundtruth)
                train_losses.append(self.model.current_train_loss)
                train_accuracies.append(self.model.current_train_accuracy)
                print(str(self.model.current_train_loss) + ' : ' + \
                    str(np.mean(train_losses)) + ' : ' + \
                    str(self.model.current_train_accuracy) + ' : ' + \
                    str(np.mean(train_accuracies)))
                infer_losses.append(self.model.current_infer_loss)
                infer_accuracies.append(self.model.current_infer_accuracy)
                print(str(self.model.current_infer_loss) + ' : ' + \
                    str(np.mean(infer_losses)) + ' : ' + \
                    str(self.model.current_infer_accuracy) + ' : ' + \
                    str(np.mean(infer_accuracies)))

                '''if self.model.current_train_accuracy > 0.95:
                    return np.mean(train_losses), np.mean(train_accuracies), \
                        np.mean(infer_losses), np.mean(infer_accuracies)'''

                if phase != 'train':
                    label_pred = ''
                    label_gold = ''
                    for batch in range(minigroundtruth.shape[0]):
                        all_were_correct = 1
                        for token in range(self.model.current_infer_prediction.shape[1]):
                            if self.model.current_infer_prediction[batch][token] == \
                                    self.model.num_classes - 1 and \
                                    minigroundtruth[batch][token] == \
                                    self.model.num_classes - 1:
                                break
                            if self.model.current_infer_prediction[batch][token] != \
                                    minigroundtruth[batch][token]:
                                all_were_correct = 0
                                break
                        if self.model.current_infer_prediction.shape[1] < \
                                minigroundtruth.shape[1] and \
                                minigroundtruth[batch] \
                                [self.model.current_infer_prediction.shape[1]]\
                                != self.model.num_classes - 1:
                            all_were_correct = 0
                        num_samples = num_samples + 1
                        samples_correct = samples_correct + all_were_correct
                        if self.model.used_loss == 'label':
                            line = minibatch_imgnames[batch] + '\t'
                            label_pred = ''
                            label_gold = ''
                            for token in range(self.model.current_infer_prediction.shape[1]):
                                if self.model.num_classes - 1 == \
                                        self.model.current_infer_prediction[batch][token]:
                                    break
                                label_pred = label_pred + self.model.vocabulary[ \
                                    self.model.current_infer_prediction[batch][token]] + ' '
                                #code.interact(local=dict(globals(), **locals()))
                            for token in range(minigroundtruth.shape[1]):
                                if self.model.num_classes - 1 == \
                                        int(minigroundtruth[batch][token]):
                                    break
                                label_gold = label_gold + self.model.vocabulary[ \
                                    int(minigroundtruth[batch][token])] + ' '                        
                            line = line + label_gold[:-1] + '\t' + label_pred[:-1] + '\t' + \
                                '-1' + '\t' + '-1' + '\n'
                            lines = lines + line
                        '''else:
                            for token in range(minigroundtruth.shape[1]):
                                if minigroundtruth[batch][token] == 0 and \
                                        minibatch_predictions[batch][token] == 0:
                                    false2false[token] = false2false[token] + 1
                                elif minigroundtruth[batch][token] == 0 and \
                                        minibatch_predictions[batch][token] == 1:
                                    false2true[token] = false2true[token] + 1
                                elif minigroundtruth[batch][token] == 1 and \
                                        minibatch_predictions[batch][token] == 1:
                                    true2true[token] = true2true[token] + 1
                                elif minigroundtruth[batch][token] == 1 and \
                                        minibatch_predictions[batch][token] == 0:
                                    true2false[token] = true2false[token] + 1
                                else:
                                    print('error!!!')
                                    code.interact(local=dict(globals(), **locals()))'''
                    if self.model.used_loss == 'label':
                        print('')
                        print(label_gold)
                        print('')
                        print(label_pred)
                        print('')
            else:
                self.model.writeParamInList('train/losses_' + 
                    str(self.model.current_epoch), str(np.mean(train_losses)))
                self.model.writeParamInList('train/accuracies_' + 
                    str(self.model.current_epoch), str(np.mean(train_accuracies)))
                self.model.writeParamInList('train/infer_losses_' + \
                    str(self.model.current_epoch), str(np.mean(infer_losses)))
                self.model.writeParamInList('train/infer_accuracies_' + \
                    str(self.model.current_epoch), str(np.mean(infer_accuracies)))
            batch_images, minibatchsize, batch_it, \
                groundtruth, new_iteration, batch_classes_true, \
                batch_imgnames, batch_labels = self.__loadBatch(phase, batch_it)
        if phase != 'train':
            self.model.writeParam(phase + '_results/epoch' + \
                str(self.model.current_epoch), lines)
            absolute_acc = float(samples_correct) / float(num_samples)
            self.model.writeParam(phase + '_absolute_accuracy/epoch' + \
                str(self.model.current_epoch), str(absolute_acc))
            self.model.writeParam(phase + '_token_accuracy/epoch' + \
                str(self.model.current_epoch), str(np.mean(infer_accuracies)))
            print('abs: ' + str(absolute_acc) + ', tok: ' + str(np.mean(infer_accuracies)))
            '''if self.model.used_loss == 'classes':
                with open(self.model.model_dir + '/confusion.npz', 'w') as fout:
                    np.savez(fout,false2false=false2false,false2true=false2true, \
                        true2true=true2true,true2false=true2false)
                stats = {}
                for c in range(self.model.num_classes):
                    num_right = float(false2false[c] + true2true[c])
                    num = float(false2false[c] + true2true[c] + false2true[c] + true2false[c])
                    n = true2true[c] + true2false[c]
                    class_accuracy = num_right / num
                    f2f = false2false[c] / num
                    t2t = true2true[c] / num
                    f2t = false2true[c] / num
                    t2f = true2false[c] / num
                    stats.update({(class_accuracy,c):(self.model.vocabulary[c], str(f2f), \
                        str(t2t), str(f2t), str(t2f), str(num))})
                keys = sorted(stats.keys())
                s = ''
                for key in keys:
                    c, f2f, t2t, f2t, t2f, n = stats[key]
                    s = s +'\"'+c+'\" : '+str(key[0])+' : ('+f2f+','+t2t+','+f2t+','+ \
                        t2f+') : ' + n + '\n'
                self.model.writeParam(phase + 'stats/epoch' + \
                    str(self.model.current_epoch), absolute_acc)'''
        return np.mean(train_losses), np.mean(train_accuracies), \
            np.mean(infer_losses), np.mean(infer_accuracies)

    def loadModel(self, model_dir=None, feature_extractor='', \
            encoder='', decoder='', encoder_size='', decoder_size='',
            optimizer='', max_num_tokens=150, only_inference=False):
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                 Try to restore model!                 ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        if model_dir == None:
            model_dir = self.findBestModel(feature_extractor, encoder, decoder,\
                encoder_size, decoder_size, optimizer)
        print(model_dir)
        #if model_dir.split('/')[-1][0] == '_':
        #    code.interact(local=dict(globals(), **locals()))
        self.model = Model(model_dir, max_num_tokens=max_num_tokens, loaded=True,
            only_inference=only_inference)
        print('load variables!')
        if self.best:
            self.model.restoreBestCheckpoint()
        else:
            self.model.restoreLastCheckpoint()
        if not only_inference:
            self.model.createOptimizer()
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                    Model restored!                    ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')

        
    def combineModels(self, fe_dir=None, ed_dir=None, feature_extractor='', \
            encoder='', decoder='', encoder_size='', decoder_size='',
            optimizer='', max_num_tokens=None):
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                Try to combine models!                 ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        if fe_dir == None:
            fe_dir = self.findBestModel(feature_extractor, optimizer=optimizer)
        variables = {}
        self.loadModel(fe_dir)
        feature_extractor = self.model.feature_extractor
        for v in tf.global_variables():
            value = self.model.session.run(v)
            variables.update({str(v.name): value})
        #if ed_dir == None:
        #    ed_dir = self.findBestModel(feature_extractor, optimizer=optimizer)
        self.loadModel(ed_dir)
        for v in tf.global_variables():
            value = self.model.session.run(v)
            variables.update({str(v.name): value})
        #code.interact(local=dict(globals(), **locals()))
        self.model = Model(self.model_dir, feature_extractor=feature_extractor, \
            encoder=self.model.encoder, decoder=self.model.decoder, \
            encoder_size=self.model.decoder_size, \
            decoder_size=self.model.decoder_size, \
            optimizer=self.model.optimizer, vocabulary=self.vocabulary, \
            max_num_tokens=self.model.max_num_tokens, \
            capacity = self.capacity)
        for v in tf.trainable_variables():
            if v.name in variables.keys():
                v = v.assign(variables[v.name])
                x = self.model.session.run(v)
        self.model.printGlobalVariables()
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                   Models combined!                    ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')

    def testModel(self):
        self.best = True
        self.loadModel(self.model.model_dir, only_inference=True)
        train_loss, train_accuracy, infer_loss, test_infer_accuracy = \
            self.__iterateOneEpoch('test')
        print('model testing is finished!')
        path = self.tmp_dir + '/' + feature_extractor + '_' + encoder + '_' + decoder + '_' +\
            encoder_size + '_' + decoder_size + '_' + optimizer
        if not os.path.exists(path):
            self.writeLog(path, model_dir)
        # code.interact(local=dict(globals(), **locals()))

    '''def renderImages(self, model_dir='', phase='test'):
        if model_dir == '':
            model_dir = self.model.model_dir
            epoch = self.model.current_epoch
        else:
            epoch = int(self.readLog(model_dir + '/params/current_epoch'))
        if phase == 'val':
            epoch = epoch - 1
        result_path = model_dir + '/params/'+ phase + '_results/epoch' + str(epoch)
        self.renderOutput(result_path)

    def renderOutput(self, result_path):
        my_renderer.render_output(result_path)

    def evaluate(self):
        decoder = self.readLog(model_dir + '/params/decoder')
        encoder_size = self.readLog(model_dir + '/params/encoder_size')
        decoder_size = self.readLog(model_dir + '/params/decoder_size')
        optimizer = self.readLog(model_dir + '/params/optimizer')
        best_model_path = self.__findBestModel(feature_extractor, encoder, \
            decoder, encoder_size, optimizer)
        if best_model_path != None:
            best_val_text_edit_distance = \
                float(self.readLog(best_model_path+'/val_text_edit_distance'))
            if val_text_edit_distance < best_val_text_edit_distance:
                return val_text_edit_distance
        #path = self.tmp_dir + '/' + self.preprocessing + self.model.getTrainMode() + '_' \
        #    + self.mode
        path = self.tmp_dir + '/' + feature_extractor + '_' + encoder + '_' + decoder + '_' +\
            encoder_size + '_' + decoder_size + '_' + optimizer
        self.writeLog(path, model_dir)
        return val_text_edit_distance

    def evaluateFormulas(self, model_dir, result_path):
        token_accuracy = self.calculateTokenAccuracy(result_path)
        print('token accuracy: ' + str(token_accuracy))
        self.writeLog(model_dir + '/token_accuracy', token_accuracy)
        abs_accuracy = self.calculateAbsoluteAccuracy(result_path)
        print('absolute accuracy: ' + str(abs_accuracy))
        self.writeLog(model_dir + '/abs_accuracy', abs_accuracy)
        text_edit_distance = self.calculateEditDistance(result_path)
        print('text edit distance: ' + str(text_edit_distance))
        self.writeLog(model_dir + '/val_text_edit_distance', val_text_edit_distance)

    def evaluateImages(self, model_dir, img_dir):
        image_edit_distance, image_accuracy = iedc.calcImageEditDistance(img_dir)
        print('image edit distance: ' + str(image_edit_distance))
        print('image accuracy: ' + str(image_accuracy))
        self.writeLog(model_dir + '/image_edit_distance', image_edit_distance)
        self.writeLog(model_dir + '/image_accuracy', image_accuracy)'''

    def readLog(self, path):
        reader = open(path,'r')
        value = reader.read()
        reader.close()
        return value

    def writeLog(self, path, value):
        writer = open(path,'w')
        value = writer.write(str(value))
        writer.close()

    def calculateEditDistance(self, result_file):
        total_ref = 0
        total_edit_distance = 0
        with open(result_file) as fin:
            for idx,line in enumerate(fin):
                if idx % 100 == 0:
                    print (idx)
                items = line.strip().split('\t')
                if len(items) == 5:
                    img_path, label_gold, label_pred, score_pred, score_gold = items
                    l_pred = label_pred.strip()
                    l_gold = label_gold.strip()
                    tokens_pred = l_pred.split(' ')
                    tokens_gold = l_gold.split(' ')
                    ref = max(len(tokens_gold), len(tokens_pred))
                    edit_distance = distance.levenshtein(tokens_gold, tokens_pred)
                    total_ref += ref
                    total_edit_distance += edit_distance
        return 1.0 - float(total_edit_distance) / total_ref

    def __findBestModel(self, feature_extractor='', encoder='', decoder='', \
        encoder_size=0, decoder_size=0, optimizer=''):
        path = self.tmp_dir + '/' + feature_extractor + '_' + encoder + '_' + decoder \
            + '_' + str(encoder_size) + '_' + str(decoder_size) + '_' + optimizer + \
            '_' + self.mode
        if not os.path.exists(path):
            return None
        reader = open(path, 'r')
        best_model_path = reader.read()
        reader.close()
        return best_model_path

    def __getBatchDir(self, phase): # vorsicht bei encdecOnly!!!
        return self.dataset_dir + '/' + self.preprocessing + phase + '_batches'

    def __getMaxNumTokens(self, path):
        batch0 = np.load(path + '/' + os.listdir(path)[0])
        return len(batch0['labels'][-1])

    def __getBatchNames(self, phase):
        assert os.path.exists(self.__getBatchDir(phase)), \
            self.__getBatchDir(phase) + " directory doesn't exists!"
        batchnames = os.listdir(self.__getBatchDir(phase))
        assert batchnames != [], self.__getBatchDir(phase)+" batch directory musn't be empty!"
        return batchnames

    def __loadBatch(self, phase, batch_it):
        batch = None
        batch_images = None
        new_iteration = False
        minibatchsize = None
        batch_it_old = batch_it
        batch_names = self.__getBatchNames(phase)
        #code.interact(local=dict(globals(), **locals()))
        while True:
            batch = np.load(self.__getBatchDir(phase) + '/' + batch_names[batch_it])
            batch_images = batch['images']
            batch_imgnames = batch['image_names']
            #code.interact(local=dict(globals(), **locals()))
            minibatchsize = self.__calculateMinibatchsize(batch_images[0])
            if minibatchsize != 0:
                break
            else:
                print('Pictures to big to process!')
                new_iteration = (batch_it + 1 >= len(batch_names))
                batch_it = (batch_it + 1) % len(batch_names)
                assert batch_it != batch_it_old, 'capacity to small for training data'
        classes_true = batch['contained_classes_list']
        labels = batch['labels']
        if self.model.used_loss == 'label':
            groundtruth = batch['labels']
        elif self.model.used_loss == 'classes':
            groundtruth = batch['contained_classes_list']
        new_iteration = (batch_it + 1 >= len(batch_names))
        batch_it = (batch_it + 1) % len(batch_names)
        assert len(batch_images) == len(groundtruth) != 0
        return batch_images, minibatchsize, batch_it, groundtruth, new_iteration, \
            classes_true, batch_imgnames, labels

    def __calculateMinibatchsize(self, image):
        minibatchsize = 20
        while image.shape[0] * image.shape[1] * image.shape[2] * minibatchsize > \
                self.model.capacity:
            minibatchsize = minibatchsize - 1
        # code.interact(local=dict(globals(), **locals()))
        return minibatchsize

    def __createMinibatch(self, batch_data, minibatch_it, minibatchsize):
        minibatch_data = []
        for it in range(minibatch_it * minibatchsize, min((minibatch_it + 1) * \
                minibatchsize,len(batch_data))):
            minibatch_data.append(batch_data[it])
        return np.array(minibatch_data)

    def saveLossGraph(self):
        train_train_loss_strings = self.model.readParamList('stats/losses/train_train')
        train_infer_loss_strings = self.model.readParamList('stats/losses/train_infer')
        val_train_loss_strings = self.model.readParamList('stats/losses/val_train')
        val_infer_loss_strings = self.model.readParamList('stats/losses/val_infer')
        train_train_losses = []
        train_infer_losses = []
        val_train_losses = []
        val_infer_losses = []
        for i in range(len(train_train_loss_strings)):
            train_train_losses.append(float(train_train_loss_strings[i]))
            train_infer_losses.append(float(train_infer_loss_strings[i]))
            val_train_losses.append(float(val_train_loss_strings[i]))
            val_infer_losses.append(float(val_infer_loss_strings[i]))
        plt.plot(range(1,len(train_train_losses) + 1), train_train_losses, 'b--')
        plt.plot(range(1,len(train_infer_losses) + 1), train_infer_losses, 'blue')
        plt.plot(range(1,len(val_train_losses) + 1), val_train_losses, 'r--')
        plt.plot(range(1,len(val_infer_losses) + 1), val_infer_losses, 'red')
        save_path = self.model.model_dir + '/plots/losses'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/epoch' + str(self.model.current_epoch))
        plt.close()

    def saveAccGraph(self):
        train_train_accs_strings = self.model.readParamList('stats/accs/train_train')
        train_infer_accs_strings = self.model.readParamList('stats/accs/train_infer')
        val_train_accs_strings = self.model.readParamList('stats/accs/val_train')
        val_infer_accs_strings = self.model.readParamList('stats/accs/val_infer')
        train_train_accs = []
        train_infer_accs = []
        val_train_accs = []
        val_infer_accs = []
        for i in range(len(train_train_accs_strings)):
            train_train_accs.append(float(train_train_accs_strings[i]))
            train_infer_accs.append(float(train_infer_accs_strings[i]))
            val_train_accs.append(float(val_train_accs_strings[i]))
            val_infer_accs.append(float(val_infer_accs_strings[i]))
        plt.plot(range(1,len(train_train_accs) + 1), train_train_accs, 'b--')
        plt.plot(range(1,len(train_infer_accs) + 1), train_infer_accs, 'blue')
        plt.plot(range(1,len(val_train_accs) + 1), val_train_accs, 'r--')
        plt.plot(range(1,len(val_infer_accs) + 1), val_infer_accs, 'red')
        save_path = self.model.model_dir + '/plots/accs'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/epoch' + str(self.model.current_epoch))
        plt.close()

    def saveTrainLossGraph(self):
        train_losses = []
        infer_losses = []
        for epoch in range(self.model.current_epoch):
            train_loss_strings = self.model.readParamList('train/losses_' + str(epoch))
            infer_loss_strings = self.model.readParamList('train/infer_losses_' + str(epoch))
            for i in range(len(train_loss_strings)):
                train_losses.append(float(train_loss_strings[i]))
                infer_losses.append(float(infer_loss_strings[i]))
        plt.plot(range(1,len(train_losses) + 1), train_losses, 'b--')
        plt.plot(range(1,len(infer_losses) + 1), infer_losses, 'blue')
        save_path = self.model.model_dir + '/plots/train_losses'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/epoch' + str(self.model.current_epoch))
        plt.close()

    def saveTrainAccGraph(self):
        train_accs = []
        infer_accs = []
        for epoch in range(self.model.current_epoch):
            train_accs_strings = self.model.readParamList('train/accs_' + str(epoch))
            infer_accs_strings = self.model.readParamList('train/infer_accs_' + str(epoch))
            for i in range(len(train_accs_strings)):
                train_accs.append(float(train_accs_strings[i]))
                infer_accs.append(float(infer_accs_strings[i]))
        plt.plot(range(1,len(train_accs) + 1), train_accs, 'b--')
        plt.plot(range(1,len(infer_accs) + 1), infer_accs, 'blue')
        save_path = self.model.model_dir + '/plots/train_accs'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/epoch' + str(self.model.current_epoch))
        plt.close()



