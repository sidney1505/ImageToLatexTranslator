import sys, os
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['xtick.labelsize'] = 7
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import difflib
import numpy as np
import distance
import code
import shutil
# own packages
from myutils.LevSeq import StringMatcher
from myutils.render_output import render_output

class DatasetAnalyser:
    def __init__(self, dataset_dir, vocab_path=None):
        if vocab_path == None:
            vocab_path = dataset_dir + '/vocabulary.txt'
        self.vocabulary = readParamList(vocab_path)
        self.dataset_dir = dataset_dir

    def analyseTokennumber(self):
        vocabulary = self.vocabulary
        endtoken = len(vocabulary)
        #
        numbers_train = np.zeros(626)
        # code.interact(local=dict(globals(), **locals()))
        for batch_name in self.__getBatchNames('train'):
            print('train' + '_' + batch_name + ' loaded!')
            batch = np.load(self.__getBatchDir('train') + '/' + batch_name)
            labels = batch['labels']
            for batch_nr in range(labels.shape[0]):
                number = 0
                for token in range(labels.shape[1]):
                    if labels[batch_nr][token] == endtoken:
                        break
                    else:
                        number = number + 1
                numbers_train[number] = numbers_train[number] + 1
        numberlist = []
        for i in range(1,len(numbers_train)):
            numberlist = np.append(numberlist, np.tile(i,int(numbers_train[i])))
        std_writer = open(self.dataset_dir + '/train_label_length_median.txt', 'w')
        std_writer.write(str(np.median(numberlist)))
        std_writer.close()
        #
        numbers_test = np.zeros(626)
        for batch_name in self.__getBatchNames('test'):
            print('test' + '_' + batch_name + ' loaded!')
            batch = np.load(self.__getBatchDir('test') + '/' + batch_name)
            labels = batch['labels']
            for batch_nr in range(labels.shape[0]):
                number = 0
                for token in range(labels.shape[1]):
                    if labels[batch_nr][token] == endtoken:
                        break
                    else:
                        number = number + 1
                numbers_test[number] = numbers_test[number] + 1
        numberlist = []
        for i in range(1,len(numbers_test)):
            numberlist = np.append(numberlist, np.tile(i,int(numbers_test[i])))
        std_writer = open(self.dataset_dir + '/test_label_length_median.txt', 'w')
        std_writer.write(str(np.median(numberlist)))
        std_writer.close()
        #
        bins = [15,30,45,60,75,90,105,120,135,151,626]
        bin_labels = ['<15','[15,30)','[30,45)','[45,60)','[60,75)','[75,90)','[90,105)', \
            '[105,120)','[120,135)', '[135,150]', '>150']
        binc_train = np.zeros(len(bins))
        binc_test = np.zeros(len(bins))
        c = 1
        for i in range(len(bins)):
            for j in range(c,bins[i]):
                #print(str(i) + ',' + str(j))
                binc_train[i] = binc_train[i] + numbers_train[j]
                binc_test[i] = binc_test[i] + numbers_test[j]
            c = bins[i]
        binfreq_train = binc_train / np.sum(binc_train)
        binfreq_test = binc_test / np.sum(binc_test)
        x_pos = np.array(range(len(bins)))
        #code.interact(local=dict(globals(), **locals()))
        fig, ax = plt.subplots()
        width = 0.35
        rects1 = ax.bar(x_pos, binfreq_train * 100, width, color='b')
        rects2 = ax.bar(x_pos + width, binfreq_test * 100, width, color='r')
        ax.legend((rects1[0], rects2[0]), ('train', 'test'))
        # plt.bar(x_pos, binfreq, align='center', alpha=1.0)
        plt.xticks(x_pos + 0.5 * width, bin_labels, rotation=45)
        plt.xlabel('#Tokens')
        plt.ylabel('Frequency (%)')
        '''def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2.,
                    height + 0.5,
                    '%f' % height,
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=5)
        autolabel(rects1)
        autolabel(rects2)'''
        plt.savefig(self.dataset_dir + '/label_length_stats.png')
        plt.close()

    def analyseVocabFrequency(self):
        vocabulary = self.vocabulary
        #
        vocab_number_train = np.zeros(len(vocabulary))
        n_train = 0
        for batch_name in self.__getBatchNames('train'):
            print('train' + '_' + batch_name + ' loaded!')
            batch = np.load(self.__getBatchDir('train') + '/' + batch_name)
            labels = batch['labels']
            for batch_nr in range(labels.shape[0]):
                for token in range(len(vocab_number_train)):
                    if token in labels[batch_nr]:
                        vocab_number_train[token] += 1
                n_train += 1
        vocab_rel_number_train = vocab_number_train / float(n_train)
        #
        vocab_number_test = np.zeros(len(vocabulary))
        n_test = 0
        for batch_name in self.__getBatchNames('test'):
            print('test' + '_' + batch_name + ' loaded!')
            batch = np.load(self.__getBatchDir('test') + '/' + batch_name)
            labels = batch['labels']
            for batch_nr in range(labels.shape[0]):
                for token in range(len(vocab_number_test)):
                    if token in labels[batch_nr]:
                        vocab_number_test[token] += 1
                n_test += 1
        vocab_rel_number_test = vocab_number_test / float(n_test)
        #
        sort_idx = np.argsort(vocab_rel_number_test)[::-1]
        sorted_vocabulary = []
        sorted_vocab_numbers_train = []
        sorted_vocab_numbers_test = []
        sorted_vocab_string = ''
        for idx in range(5):
            sorted_vocabulary.append(vocabulary[sort_idx[idx]])
            sorted_vocab_numbers_train.append(vocab_rel_number_train[sort_idx[idx]])
            sorted_vocab_numbers_test.append(vocab_rel_number_test[sort_idx[idx]])
            sorted_vocab_string = sorted_vocab_string + str(vocabulary[sort_idx[idx]]) + \
                ' : ' + str(vocab_rel_number_train[sort_idx[idx]]) + '\n'
        sorted_vocabulary.append('...')
        sorted_vocab_numbers_train.append(0)
        sorted_vocab_numbers_test.append(0)
        sidx = len(vocab_rel_number_test)
        for idx in range(1,len(vocab_rel_number_test)):
            if vocab_rel_number_test[sort_idx[-idx]] > 0:
                sidx = idx
                break
        for idx in range(sidx,sidx + 5)[::-1]:
            sorted_vocabulary.append(vocabulary[sort_idx[-idx]])
            sorted_vocab_numbers_train.append(vocab_rel_number_train[sort_idx[-idx]])
            sorted_vocab_numbers_test.append(vocab_rel_number_test[sort_idx[-idx]])
            sorted_vocab_string = sorted_vocab_string + str(vocabulary[sort_idx[-idx]]) + \
                ' : ' + str(vocab_rel_number_train[sort_idx[-idx]]) + '\n'
        #
        vocab_writer = open(self.dataset_dir + '/tokenclass_freq.txt', 'w')
        vocab_writer.write(sorted_vocab_string)
        vocab_writer.close()
        #
        #svocabs = sorted_vocabulary[1:21]
        # TODO!!!!
        x_pos = np.arange(len(sorted_vocabulary))
        # snumbers = sorted_vocab_numbers[1:21] ???
        fig, ax = plt.subplots()
        width = 0.35
        rects1 = ax.bar(x_pos, np.array(sorted_vocab_numbers_train) * 100, width, color='b')
        rects2 = ax.bar(x_pos + width, np.array(sorted_vocab_numbers_test) * 100, width, color='r')
        ax.legend((rects1[0], rects2[0]), ('train', 'test'))
        # plt.bar(x_pos, snumbers, align='center', alpha=0.5)
        plt.xticks(x_pos + width / 2, sorted_vocabulary, rotation=90)
        plt.xlabel('Token Class')
        plt.ylabel('Frequency (%)')
        def autolabel(rects):
            for i in range(11,len(rects)):
                height = rects[i].get_height()
                ax.text(rects[i].get_x() + rects[i].get_width()/2.,
                    height + 0.5,
                    '%f' % height,
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=5)
        autolabel(rects1)
        autolabel(rects2)
        plt.savefig(self.dataset_dir + '/vocab_freq.png')
        plt.close()

    def __getBatchDir(self, phase): # vorsicht bei encdecOnly!!!
        return self.dataset_dir + '/' + phase + '_batches'

    def __getBatchNames(self, phase):
        assert os.path.exists(self.__getBatchDir(phase)), \
            self.__getBatchDir(phase) + " directory doesn't exists!"
        batchnames = os.listdir(self.__getBatchDir(phase))
        assert batchnames != [], self.__getBatchDir(phase)+" batch directory musn't be empty!"
        return batchnames

def readParamList(read_path):
    if not os.path.exists(read_path):
        print(read_path + ' does not exist!')
        return []
    reader = open(read_path, 'r')
    value = reader.read().split('\n')
    reader.close()
    return value

def main():
    print('enter main method')
    da = DatasetAnalyser(os.environ['DATA_DIR'])
    da.analyseTokennumber()
    da.analyseVocabFrequency()

if __name__ == '__main__':
    main()