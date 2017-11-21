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
# own packages
import myutils.render_output as my_renderer
import myutils.ImageEditDistanceCalculator as iedc

class Evaluator:
    def __init__(self, model_dir, phase='test'):
        self.model_dir = model_dir
        self.phase = phase
        self.evaluation_path = self.model_dir + '/' + phase '_evaluation'

    def evaluateFormulas(self, result_path, vocabulary, max_length=625):
        counts = np.zeros(max_length)
        total_corrects = np.zeros(max_length)
        total_tokens_corrects = np.zeros(max_length)
        confusion = np.zeros([len(vocabulary),len(vocabulary)])
        total_refs = np.zeros(max_length)
        total_edit_distances = np.zeros(max_length)
        correct = 0
        with open(result_file) as fin:
            for idx,line in enumerate(fin):
                if idx % 100 == 0:
                    print (idx)
                items = line.strip().split('\t')
                if len(items) == 5:
                    img_path, label_gold, label_pred, score_pred, score_gold = items
                    if label_gold == label_pred:
                        correct = 1
                    l_pred = label_pred.strip()
                    l_gold = label_gold.strip()
                    tokens_pred = l_pred.split(' ')
                    tokens_gold = l_gold.split(' ')
                    ref = max(len(tokens_gold), len(tokens_pred))
                    ref_min = min(len(tokens_gold), len(tokens_pred))
                    length = len(tokens_gold)
                    counts[length] = counts[length] + 1
                    #
                    total_correct[length] = total_correct[length] + correct
                    #
                    tokens_correct = 0.0
                    for i in range(ref_min):
                    	if tokens_pred[i] == tokens_gold[i]:
                    		tokens_correct = tokens_correct + 1
                        elif i == 0 or tokens_pred[i-1] == tokens_gold[i-1]:
                            gold = vocabulary.index(tokens_gold[i])
                            pred = vocabulary.index(tokens_pred[i])
                            confusion[gold][pred] = confusion[gold][pred] + 1
                    total_tokens_correct = total_tokens_correct + tokens_correct / ref
                    #
                    edit_distance = distance.levenshtein(tokens_gold, tokens_pred)
                    total_ref += ref
                    total_edit_distance += edit_distance
                    correct = 0
        absolute_accuracies = total_corrects / counts
        token_accuracies = total_tokens_corrects / counts
        text_edit_distances = 1.0 - total_edit_distances / total_ref
        confusion = confusion / np.sum(confusion)
        #
        with open(self.model_dir + "evaluation/analysis.npz", 'w') as fout:
            np.savez(fout, absolute_accuracies=absolute_accuracies, \
            	token_accuracies=token_accuracies, \
                text_edit_distances=text_edit_distances, \
                confusion=confusion)
        #
        def weighted_average(metric, l=0, r=max_length):
        	akk = 0
        	total_count = 0
        	for i in range(l,r):
        		akk = akk + metric[i] * counts[i]
        		total_count = total_count + counts[i]
        	return akk / total_count
        token_accuracy = weighted_average(token_accuracies)
        absolute_accuracy = weighted_average(absolute_accuracies)
        text_edit_distance = weighted_average(total_edit_distances)
        #
        print('token accuracy: ' + str(token_accuracy))
        print('absolute accuracy: ' + str(abs_accuracy))
        print('text edit distance: ' + str(text_edit_distance))
        #
        self.writeLog(self.model_dir + '/' + self.phase + '_token_accuracy', token_accuracy)
        self.writeLog(self.model_dir + '/' + self.phase + '_abs_accuracy', abs_accuracy)
        self.writeLog(self.model_dir + '/' + self.phase + '_text_edit_distance', \
            text_edit_distance)
        # create and save bins
        bins = [0,15,30,45,60,75,90,105,120,135,max_length]
        bin_labels = ['<15','[15,30)','[30,45)','[45,60)','[60,75)','[75,90)','[90,105)', \
            '[105,120)','[120,135)', '>=135']
		def per_bin(metric):
			binc = np.zeros(len(bins) - 1)
			for i in range(0,len(bins)-1):
				binc[i] = weighted_average(metric, bins[i], bins[i+1])
			return binc
        token_accuracy_bins = per_bin(token_accuracies)
        absolute_accuracy_bins = per_bin(absolute_accuracy_bins)
        text_edit_distance_bins = per_bin(text_edit_distance)
        y_pos = np.arange(len(binc))
        def savePlot(metric, name):
            plt.bar(y_pos, metric, align='center', alpha=1.0)
            plt.xticks(y_pos, bin_labels)
            plt.ylabel('%')
            plt.savefig(self.evaluation_path + '/' + name + '.pdf')
            plt.close()
        savePlot(token_accuracy_bins,'token_accuracy_bins')
        savePlot(absolute_accuracy_bins,'absolute_accuracy_bins')
        savePlot(text_edit_distance_bins,'text_edit_distance_bins')
        # extract the most important information of the confusion matrix
        m = confusion.argsort()[::-1][:20]
        m_from = m / len(vocabulary)
        m_to = m % len(vocabulary)
        for k in range(20):
            vocab_from = vocabulary[m_from]
            vocab_to = vocabulary[m_to]
            value = str(confusion[m_from][m_to])
            line = vocab_from ' -> ' + vocab_to + ' : ' + value
            self.writeParamInList('confusion_outline.txt', line)

    def writeParamInList(self, path, value):
        write_path = self.evaluation_path + '/' + path
        writer = open(write_path, 'a')
        writer.write(str(value) + '\n')
        writer.close()

    def evaluateImages(self, img_dir):
        image_edit_distance, image_accuracy = iedc.calcImageEditDistance(img_dir)
        print(self.phase + 'image edit distance: ' + str(image_edit_distance))
        print(self.phase + 'image accuracy: ' + str(image_accuracy))
        self.writeLog(model_dir + '/' + self.phase + 'image_edit_distance', image_edit_distance)
        self.writeLog(model_dir + '/' + self.phase + 'image_accuracy', image_accuracy)

def main():
    print('enter main method')
    code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()