import sys, os, argparse, logging
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
        assert os.path.exists(self.dataset_dir), \
            "Dataset directory doesn't exists!"
        self.tmp_dir = tmp_dir
        self.capacity = capacity

    def setModelParameters(self, mode, feature_extractor, encoder, decoder, encoder_size, \
            decoder_size, optimizer):
        self.mode = mode
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_size = encoder_size
        self.decoder_size = decoder_siheze
        self.optimizer = optimizer
        self.initial_learning_rate = 0.1

    def trainModel(self):
        if self.mode == 'e2e':
            self.model = Model(self.model_dir, self.feature_extractor, \
                self.encoder, self.decoder, self.encoder_size, \
                self.ecoder_size, self.optimizer, self.initial_learning_rate, \
                self.initial_learning_rate)
            self.train()
        elif self.mode == 'stepbystep':
            self.model = Model(self.model_dir, self.feature_extractor, self.optimizer, \
                self.initial_learning_rate)
            self.train()
            self.processData()
            fe_dir = self.model.model_dir
            self.model = Model(self.model_dir, self.encoder, self.decoder, \
                self.encoder_size, self.decoder_size, self.optimizer, \
                self.initial_learning_rate)
            self.train()
            ed_dir = self.model.model_dir
            self.model = self.combineModels(fe_dir, ed_dir)
            self.train()

    def train():
        # the training loop
        train_last_loss = float('inf')
        val_last_loss = float('inf')
        while True:
            train_loss, train_accuracy = self.__iterateOneEpoch('train')
            val_loss, val_accuracy = self.__iterateOneEpoch('val')
            self.model.current_epoch = self.model.current_epoch + 1
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

    def loadModel(model_dir=None, feature_extractor='', \
            encoder='', decoder='', encoder_size='', decoder_size='',
            optimizer='', max_num_tokens=None):
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        print('###                 Try to restore model!                 ###')
        print('#############################################################')
        print('#############################################################')
        print('#############################################################')
        session = tf.Session()
        if model_dir == None:
            model_dir = self.findBestModel(feature_extractor, encoder, decoder,\
                encoder_size, decoder_size, optimizer)
        self.model = Model(model_dir, max_num_tokens=max_num_tokens, loaded=True, \
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
        final_loss, final_accuracy = self.__iterateOneEpoch('test')
        path = '/home/sbender/tmp/' + model.train_mode + '.tmp'
            #shutil.rmtree(write_path)
            if os.path.exists(path):
                reader = open(path, 'r')
                old_model_dir = reader.readlines()
                reader2 = open(old_model_dir + 'test/accuracy', 'r')
                old_final_accuracy = float(reader2.readlines())
                if old_final_accuracy < final_accuracy:
                    writer = open(path, 'w')
                    writer.write(model.model_dir)
            else:
                writer = open(path, 'w')
                writer.write(model.model_dir)
        return final_accuracy

    def processData(self):
        for phase in ['train','val','test']:
            batch_it = 0
            batch_images, minibatchsize, batch_it, batch_labels, new_iteration, \
                batch_classes_true, batch_imgnames = __loadBatch(self.__getBatchDir(phase),\
                self.__getBatchNames(phase), batch_it, self.model)
            while not new_iteration: # randomise!!!
                print(batch_it)
                minibatch_predictions = []
                for minibatch_it in range(batch_images.shape[0] \
                        / minibatchsize):
                    minibatch_images = self.__createMinibatch(batch_images, \
                        minibatch_it, minibatchsize)
                    minibatch_prediction = model.predict(model.features, minibatch_images)
                    minibatch_predictions.append(prediction)
                batch_prediction = np.concatenate(minibatch_predictions)
                path = self.dataset_dir + "/" + self.model.feature_extractor + '_' + \
                    phase + '_batches/batch' + batch_it + '.npz'
                with open(path, 'w') as fout:
                    np.savez(fout, images=batch_prediction, labels=batch['labels'], \
                        num_classes=model.num_classes, \
                        contained_classes_list=batch['contained_classes_list'],
                        image_names=batch_imgnames)
                print(phase + 'batch' + str(batch_it) + " saved! " + str(preds.shape) \
                    + " : " + str(batch['labels'].shape))
                batch_images, minibatchsize, batch_it, batch_labels, new_iteration, \
                    batch_classes_true, batch_imgnames = __loadBatch( \
                    self.__getBatchDir(phase), self.__getBatchNames(phase), batch_it, \
                    self.model)

    def __getBatchDir(self, phase): # vorsicht bei encdecOnly!!!
        return self.dataset_dir + '/' + phase + '_batches'

    def __getBatchNames(self, phase):
        assert os.path.exists(self.dataset_dir + '/' + phase + '_batches'), \
            phase + " directory doesn't exists!"
        batchnames = os.listdir(self.dataset_dir + '/' + phase + '_batches')
        assert batchnames != [], phase + " batch directory musn't be empty!"
        return batchnames

    def __findBestModel(self, feature_extractor='', encoder='', decoder='', encoder_size='', \
            decoder_size='', optimizer=''):
        path = self.tmp_dir + '/' + feature_extractor + '_' + encoder + '_' + decoder \
             + '_' + str(encoder_size) + '_' + str(decoder_size) + '_' + optimizer
        if not os.path.exists(path):
            print(path + ' does not exist!')
            quit()
        reader = open(path, 'r')
        best_model_path = reader.read()
        reader.close()
        return best_model_path

    def __loadBatch(self, mode, batch_it):
        batch = None
        batch_images = None
        new_iteration = False
        minibatchsize = None
        batch_labels = None
        batch_it_old = batch_it
        batch_names = self.__getBatchNames(mode)
        #code.interact(local=dict(globals(), **locals()))
        while True:
            batch = np.load(batch_dir + '/' + batch_names[batch_it])
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

    def __iterateOneEpoch(self, phase):
        if self.model.used_loss == 'label':
            results = self.model.model_dir + '/' + mode + str(self.model.current_epoch) + '_results.txt'
            shutil.rmtree(results, ignore_errors=True)
            fout = open(results, 'w')
        else:
            false2false = np.zeros(self.model.num_classes)
            false2true = np.zeros(self.model.num_classes)
            true2true = np.zeros(self.model.num_classes)
            true2false = np.zeros(self.model.num_classes)
        batch_it = 0
        batch_images, minibatchsize, batch_it, \
            batch_labels, new_iteration, batch_classes_true, \
            batch_imgnames = __loadBatch(self.__getBatchDir(phase), self.__getBatchNames(phase), \
            batch_it, self.model)
        num_samples = 0
        samples_correct = 0
        tokens_correct = 0
        num_tokens = 0
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
                    minibatch_loss_value, minibatch_accuracy,minibatch_predictions = \
                        self.model.trainStep(minibatch_images, minibatch_labels)
                else:                
                    minibatch_loss_value, minibatch_accuracy,minibatch_predictions = \
                        self.model.valStep(minibatch_images, minibatch_labels)

                print(mode + 'minibatch(' + str(epoch) + ',' + str(batch_it) \
                    + ',' + str(minibatch_it*minibatchsize)+') : '+\
                    str(minibatch_images.shape) + ' : ' + \
                    str(minibatch_labels.shape) + ' : ' + \
                    str(minibatch_loss_value) + ' : ' + \
                    str(minibatch_accuracy) + ' : ' + \
                    str(model.current_train_accuracy) + ' : ' + \
                    str(model.current_infer_accuracy) + ' : ' + \
                    self.model.train_mode)
                batch_losses.append(minibatch_loss_value)
                batch_accuracies.append(minibatch_accuracy)

                if phase != 'train':
                    for batch in range(minibatch_labels.shape[0]):
                        num_samples = num_samples + 1
                        all_were_correct = 1
                        for token in range(minibatch_labels.shape[1]):
                            num_tokens = num_tokens + 1
                            if minibatch_predictions[batch][token] != \
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
                                if len(self.model.vocabulary) == \
                                        minibatch_predictions[batch][token]+1:
                                    break
                                label_pred = label_pred + self.model.vocabulary[ \
                                    minibatch_predictions[batch][token]] + ' '
                                #code.interact(local=dict(globals(), **locals()))
                            for token in range(minibatch_labels.shape[1]):
                                if len(self.model.vocabulary)== \
                                    int(minibatch_labels[batch][token])+1:
                                    break
                                label_gold = label_gold + self.model.vocabulary[ \
                                    int(minibatch_labels[batch][token])] + ' '                        
                            line = line + label_gold[:-1] + '\t' + label_pred[:-1] + '\t' + \
                                '-1' + '\t' + '-1' + '\n'
                            print(self.model.train_mode)
                            print(label_gold[:-1])
                            print('')
                            print(label_pred[:-1])
                            print('')
                            print(str(samples_correct) + ' / ' + str(num_samples))
                            fout.write(line)
                        else:
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
                                    code.interact(local=dict(globals(), **locals()))
                        '''if batch % 10 == 0:
                            absolute_acc = float(samples_correct) / float(num_samples)
                            token_acc = float(tokens_correct) / float(num_tokens)
                            print('abs: ' + str(absolute_acc) + ', tok: ' + str(token_acc))'''
            batch_images, minibatchsize, batch_it, \
                batch_labels, new_iteration, batch_classes_true, \
                batch_imgnames = __loadBatch(self.__getBatchDir(phase), self.__getBatchNames(phase), \
                batch_it, self.model, self.model.capacity)
        if phase != train:
            absolute_acc = float(samples_correct) / float(num_samples)
            absolute_acc_path = self.model.model_dir + '/' + mode + str(self.model.current_epoch) + \
                '_absolute_accuracy.txt'
            shutil.rmtree(absolute_acc_path, ignore_errors=True)
            absolute_acc_writer = open(absolute_acc_path, 'w')
            absolute_acc_writer.write(str(absolute_acc))
            absolute_acc_writer.close()
            token_acc = float(tokens_correct) / float(num_tokens)
            token_acc_path = self.model.model_dir + '/' + mode + str(self.model.current_epoch) + \
                '_token_accuracy.txt'
            shutil.rmtree(token_acc_path, ignore_errors=True)
            token_acc_writer = open(token_acc_path, 'w')
            token_acc_writer.write(str(token_acc))
            token_acc_writer.close()
            print('abs: ' + str(absolute_acc) + ', tok: ' + str(token_acc))
            if self.model.used_loss == 'label':
                fout.close()
            else:
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
                stats_path = self.model.model_dir + '/stats.txt'
                shutil.rmtree(stats_path, ignore_errors=True)
                stats_writer = open(stats_path, 'w')
                stats_writer.write(s)
                stats_writer.close()
        return np.mean(batch_losses), np.mean(batch_accuracies)

    def __calculateMinibatchsize(self, image):
        minibatchsize = 20
        while image.shape[0] * image.shape[1] * image.shape[2] * minibatchsize > \
                model.capacity:
            minibatchsize = minibatchsize - 1
        return minibatchsize

    def __createMinibatch(self, batch_data, minibatch_it, minibatchsize):
        minibatch_data = []
        for it in range(minibatch_it * minibatchsize, min((minibatch_it + 1) * \
                minibatchsize,len(batch_data))):
            minibatch_data.append(batch_data[it])
        return np.array(minibatch_data)