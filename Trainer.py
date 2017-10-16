import sys, os, argparse, logging, time
import matplotlib.pyplot as plt
import numpy as np
import code
import shutil
import datetime
# from model.model import Model
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from models.ModelFactory import Model
import models.ModelFactory

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
        self.capacity = capacity
        # standart values for the hyperparameters
        self.mode = 'stepbystep'
        self.feature_extractor = 'wysiwygFe'
        self.encoder = 'birowEnc'
        self.decoder = 'simpleDec'
        self.encoder_size = 1024
        self.decoder_size = 1024
        self.optimizer = 'sgd'
        self.initial_learning_rate = 0.1

    def setModelParameters(self, mode, feature_extractor, encoder, decoder, encoder_size, \
            decoder_size, optimizer):
        self.mode = mode
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.optimizer = optimizer

    def trainModel(self):
        if self.mode == 'e2e':
            self.model = Model(self.model_dir, self.feature_extractor, \
                self.encoder, self.decoder, self.encoder_size, \
                self.decoder_size, self.optimizer, self.initial_learning_rate, \
                self.vocabulary, self.__getMaxNumTokens(self.__getBatchDir('train')), \
                self.capacity)
            self.train()
        elif self.mode == 'stepbystep':
            fe_dir = self.__findBestModel(self.feature_extractor, optimizer=self.optimizer)
            if fe_dir == None:
                self.model = Model(self.model_dir, self.feature_extractor, optimizer= \
                    self.optimizer, vocabulary=self.vocabulary, capacity=self.capacity, \
                    max_num_tokens =self.__getMaxNumTokens(self.__getBatchDir('train')), \
                    learning_rate=self.initial_learning_rate)
                self.train()
                self.testModel()
                fe_dir = self.model.model_dir
            else:
                self.loadModel(fe_dir)
            basedir = self.dataset_dir + "/" + self.model.feature_extractor
            if not os.path.exists(basedir + '_train_batches') or \
                    not os.path.exists(basedir + '_val_batches') or \
                    not os.path.exists(basedir + '_test_batches'):
                self.processData()
            ed_dir = self.__findBestModel(encoder=self.encoder, decoder=self.decoder, \
                encoder_size=self.encoder_size, decoder_size=self.decoder_size, \
                optimizer=self.optimizer)
            if ed_dir == None:
                self.model = Model(self.model_dir, encoder=self.encoder, \
                    decoder=self.decoder, encoder_size=self.encoder_size, \
                    decoder_size=self.decoder_size, optimizer=self.optimizer, \
                    vocabulary=self.vocabulary, capacity=self.capacity, \
                    max_num_tokens =self.__getMaxNumTokens(self.__getBatchDir('train')), \
                    learning_rate=self.initial_learning_rate)
                self.train()
                self.testModel()
                ed_dir = self.model.model_dir
            else:
                self.loadModel(ed_dir)
            self.model = self.combineModels(fe_dir, ed_dir)
            self.train()

    def processData(self):
        for phase in ['train','val','test']:
            path = self.dataset_dir + "/" + self.model.feature_extractor + '_' + \
                    phase + '_batches'
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path)
            batch_it = 0
            batch_images, minibatchsize, batch_it, batch_labels, new_iteration, \
                batch_classes_true, batch_imgnames = self.__loadBatch(phase, batch_it)
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
                batch_images, minibatchsize, batch_it, batch_labels, new_iteration, \
                    batch_classes_true, batch_imgnames = self.__loadBatch(phase, batch_it)

    def train(self):
        # the training loop
        train_last_loss = float('inf')
        val_last_loss = float('inf')
        while True:
            begin = time.time()
            train_loss, train_accuracy, infer_loss, infer_accuracy = \
                self.__iterateOneEpoch('train')
            val_loss, val_accuracy, val_infer_loss, val_infer_accuracy = \
                self.__iterateOneEpoch('val')
            # updates the implicit parameters of the model
            self.model.current_epoch = self.model.current_epoch + 1
            self.model.writeParam('current_epoch',self.model.current_epoch)
            self.model.writeParam('current_step',self.model.current_step)
            end = time.time()
            self.model.current_millis = self.model.current_millis + (end - begin)
            self.model.writeParam('current_millis',self.model.current_millis)
            # checks whether model is still learning or overfitting
            if train_last_loss < train_loss:
                train_last_loss = train_loss
                val_last_loss = val_loss
                self.model.decayLearningRate(0.5)
            elif val_last_loss > val_loss:
                train_last_loss = train_loss
                val_last_loss = val_loss
            else:
                print('model starts to overfit!')
                print('before termination')
                code.interact(local=dict(globals(), **locals()))
                break

    def __iterateOneEpoch(self, phase):
        if self.model.used_loss == 'classes':
            false2false = np.zeros(self.model.num_classes)
            false2true = np.zeros(self.model.num_classes)
            true2true = np.zeros(self.model.num_classes)
            true2false = np.zeros(self.model.num_classes)
        batch_it = 0
        batch_images, minibatchsize, batch_it, \
            batch_labels, new_iteration, batch_classes_true, batch_imgnames = \
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
            print(batch_it)
            batch_predictions = []
            for minibatch_it in range(batch_images.shape[0] \
                    / minibatchsize):
                # create the minibatches
                minibatch_images = self.__createMinibatch(batch_images, \
                    minibatch_it, minibatchsize)
                minibatch_labels = self.__createMinibatch(batch_labels, \
                    minibatch_it, minibatchsize)
                minibatch_imgnames = self.__createMinibatch(batch_imgnames, \
                    minibatch_it, minibatchsize)
                #
                #code.interact(local=dict(globals(), **locals()))
                if phase == 'train':
                    self.model.trainStep(minibatch_images, minibatch_labels)
                else:
                    self.model.valStep(minibatch_images, minibatch_labels)

                # make the prints for the minibatch
                print(phase + 'minibatch(' + str(self.model.current_epoch) + ',' + \
                    str(batch_it - 1) + ',' + str(minibatch_it*minibatchsize)+')[' + \
                    str(self.model.current_step) + '] : '+ str(minibatch_images.shape) + \
                    ' : ' + str(minibatch_labels.shape))
                print(self.model.model_dir)
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

                if self.model.current_train_accuracy > 0.95:
                    return np.mean(train_losses), np.mean(train_accuracies), \
                        np.mean(infer_losses), np.mean(infer_accuracies)

                if phase != 'train':
                    label_pred = ''
                    label_gold = ''
                    for batch in range(minibatch_labels.shape[0]):
                        num_samples = num_samples + 1
                        all_were_correct = 1
                        for token in range(minibatch_labels.shape[1]):
                            if self.model.current_infer_prediction[batch][token] == \
                                    self.model.num_classes - 1 and \
                                    minibatch_labels[batch][token] == \
                                    self.model.num_classes - 1:
                                break
                            num_tokens = num_tokens + 1
                            if self.model.current_infer_prediction[batch][token] != \
                                    minibatch_labels[batch][token]:
                                all_were_correct = 0
                            else:
                                tokens_correct = tokens_correct + 1
                        samples_correct = samples_correct + all_were_correct
                        if self.model.used_loss == 'label':
                            line = minibatch_imgnames[batch] + '\t'
                            label_pred = ''
                            label_gold = ''
                            for token in range(minibatch_labels.shape[1]):
                                if self.model.num_classes - 1 == \
                                        self.model.current_infer_prediction[batch][token]:
                                    break
                                label_pred = label_pred + self.model.vocabulary[ \
                                    self.model.current_infer_prediction[batch][token]] + ' '
                                #code.interact(local=dict(globals(), **locals()))
                            for token in range(minibatch_labels.shape[1]):
                                if self.model.num_classes - 1 == \
                                        int(minibatch_labels[batch][token]):
                                    break
                                label_gold = label_gold + self.model.vocabulary[ \
                                    int(minibatch_labels[batch][token])] + ' '                        
                            line = line + label_gold[:-1] + '\t' + label_pred[:-1] + '\t' + \
                                '-1' + '\t' + '-1' + '\n'
                            lines = lines + line
                        '''else:
                            for token in range(minibatch_labels.shape[1]):
                                if minibatch_labels[batch][token] == 0 and \
                                        minibatch_predictions[batch][token] == 0:
                                    false2false[token] = false2false[token] + 1
                                elif minibatch_labels[batch][token] == 0 and \
                                        minibatch_predictions[batch][token] == 1:
                                    false2true[token] = false2true[token] + 1
                                elif minibatch_labels[batch][token] == 1 and \
                                        minibatch_predictions[batch][token] == 1:
                                    true2true[token] = true2true[token] + 1
                                elif minibatch_labels[batch][token] == 1 and \
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
                batch_labels, new_iteration, batch_classes_true, \
                batch_imgnames = self.__loadBatch(phase, batch_it)
        if phase != 'train':
            self.model.writeParam(phase + 'results/epoch' + \
                str(self.model.current_epoch), lines)
            absolute_acc = float(samples_correct) / float(num_samples)
            self.model.writeParam(phase + 'absolute_accuracy/epoch' + \
                str(self.model.current_epoch), str(absolute_acc))
            token_acc = float(tokens_correct) / float(num_tokens)
            self.model.writeParam(phase + 'token_accuracy/epoch' + \
                str(self.model.current_epoch), str(absolute_acc))
            print('abs: ' + str(absolute_acc) + ', tok: ' + str(token_acc))
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
            optimizer='', max_num_tokens=150):
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
        self.model = Model(model_dir, max_num_tokens=max_num_tokens, loaded=True)
        print('load variables!')
        self.model.restoreLastCheckpoint()
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
        fe = self.loadModel(fe_dir)
        for v in tf.global_variables():
            value = fe.session.run(v)
            variables.update({str(v.name): value})
        ed = self.loadModel(ed_dir)
        for v in tf.global_variables():
            value = encoder_decoder.session.run(v)
            variables.update({str(v.name): value})
        model = Model(self.model_dir, feature_extractor=fe.feature_extractor,
            encoder=ed.encoder, decoder=ed.decoder, decoder_size=ed.decoder_size,
            optimizer=ed.optimizer, vocabulary=ed.vocabulary, \
            num_classes=ed.num_classes, max_num_tokens=ed.max_num_tokens)
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

    def testModel(self):
        train_loss, train_accuracy, infer_loss, infer_accuracy = \
            self.__iterateOneEpoch('test')
        final_accuracy_writer = open(self.model.model_dir + '/final_accuracy','w')
        final_accuracy_writer.write(str(infer_accuracy))
        final_accuracy_writer.close()
        best_model_path = self.__findBestModel(self.feature_extractor, self.encoder, \
            self.decoder, self.encoder_size, self.optimizer)
        if best_model_path != None:
            best_model_reader = open(best_model_path + '/final_accuracy','r')
            best_final_accuracy = float(best_model_reader.read())
            best_model_reader.close()
            if infer_accuracy < best_final_accuracy:
                return infer_accuracy
        path = self.tmp_dir + '/' + self.model.getTrainMode()
        best_model_writer = open(path,'w')
        best_model_writer.write(self.model.model_dir)
        best_model_writer.close()
        return infer_accuracy

    def __findBestModel(self, feature_extractor='', encoder='', decoder='', \
        encoder_size=0, decoder_size=0, optimizer=''):
        path = self.tmp_dir + '/' + feature_extractor + '_' + encoder + '_' + decoder \
             + '_' + str(encoder_size) + '_' + str(decoder_size) + '_' + optimizer
        if not os.path.exists(path):
            return None
        reader = open(path, 'r')
        best_model_path = reader.read()
        reader.close()
        return best_model_path

    def __getBatchDir(self, phase): # vorsicht bei encdecOnly!!!
        return self.dataset_dir + '/' + phase + '_batches'

    def __getMaxNumTokens(self, path):
        batch0 = np.load(path + '/' + os.listdir(path)[0])
        return len(batch0['labels'][-1])

    def __getBatchNames(self, phase):
        assert os.path.exists(self.dataset_dir + '/' + phase + '_batches'), \
            phase + " directory doesn't exists!"
        batchnames = os.listdir(self.dataset_dir + '/' + phase + '_batches')
        assert batchnames != [], phase + " batch directory musn't be empty!"
        return batchnames

    def __loadBatch(self, phase, batch_it):
        batch = None
        batch_images = None
        new_iteration = False
        minibatchsize = None
        batch_labels = None
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
        if self.model.used_loss == 'label':
            batch_labels = batch['labels']
        elif self.model.used_loss == 'classes':
            batch_labels = batch['contained_classes_list']
        classes_true = batch['contained_classes_list']
        new_iteration = (batch_it + 1 >= len(batch_names))
        batch_it = (batch_it + 1) % len(batch_names)
        assert len(batch_images) == len(batch_labels) != 0
        return batch_images, minibatchsize, batch_it, batch_labels, new_iteration, \
            classes_true, batch_imgnames

    def drawLossGraph(self, epoch):
        train_loss_strings = self.model.readParamList('train/losses_' + str(epoch))
        train_losses = []
        infer_loss_strings = self.model.readParamList('train/infer_losses_' + str(epoch))
        infer_losses = []
        for i in range(len(train_loss_strings)):
            trainlosses.append(float(train_loss_strings[i]))
            inferlosses.append(float(infer_loss_strings[i]))
        plt.plot(range(1,len(train_losses) + 1), train_losses, color='blue')
        plt.plot(range(1,len(infer_losses) + 1), infer_losses, color='red')
        plt.show()

    def __calculateMinibatchsize(self, image):
        minibatchsize = 20
        while image.shape[0] * image.shape[1] * image.shape[2] * minibatchsize > \
                self.model.capacity:
            minibatchsize = minibatchsize - 1
        return minibatchsize

    def __createMinibatch(self, batch_data, minibatch_it, minibatchsize):
        minibatch_data = []
        for it in range(minibatch_it * minibatchsize, min((minibatch_it + 1) * \
                minibatchsize,len(batch_data))):
            minibatch_data.append(batch_data[it])
        return np.array(minibatch_data)