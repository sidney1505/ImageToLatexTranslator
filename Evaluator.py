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
import myutils.render_output as my_renderer

class Evaluator:
    def __init__(self, model_dir, phase='test'):
        self.model_dir = model_dir
        self.phase = phase
        self.evaluation_path = self.model_dir + '/' + phase + '_evaluation'
        shutil.rmtree(self.evaluation_path, ignore_errors=True)
        os.makedirs(self.evaluation_path)

    def evaluate(self, result_file, image_dir, vocabulary, max_length=625):
        counts = np.zeros(max_length)
        # label evaluation
        total_corrects = np.zeros(max_length)
        total_tokens_corrects = np.zeros(max_length)
        confusion = np.zeros([len(vocabulary),len(vocabulary)])
        total_refs = np.zeros(max_length)
        total_edit_distances = np.zeros(max_length)
        # image evaluation
        total_image_edit_distances = np.zeros(max_length)
        total_image_refs = np.zeros(max_length)
        total_image_corrects = np.zeros(max_length)
        total_image_correct_eliminates = np.zeros(max_length)
        #
        correct = 0
        #
        def weighted_average(metric, l=0, r=max_length):
            akk = 0
            total_count = 0
            for i in range(l,r):
                if counts[i] != 0:
                    akk += metric[i] * counts[i]
                    total_count += counts[i]
            return akk / total_count
        # code.interact(local=dict(globals(), **locals()))
        with open(result_file) as fin:
            for idx,line in enumerate(fin):
                items = line.strip().split('\t')
                if len(items) == 5:
                    img_path, label_gold, label_pred, score_pred, score_gold = items
                    filename_gold = image_dir + '/images_gold/' + img_path
                    filename_pred = image_dir + '/images_pred/' + img_path
                    if not os.path.exists(filename_gold):
                        print(filename_gold + " doesn't exists!")
                        continue
                    # label evaluation
                    if label_gold == label_pred:
                        correct = 1
                    l_pred = label_pred.strip()
                    l_gold = label_gold.strip()
                    tokens_pred = l_pred.split(' ')
                    tokens_gold = l_gold.split(' ')
                    ref = max(len(tokens_gold), len(tokens_pred))
                    ref_min = min(len(tokens_gold), len(tokens_pred))
                    if l_pred == '':
                        print('prediction label empty')
                        ref_min = 0
                    if l_gold == '':
                        print('gold label empty')
                        ref_min = 0
                    length = len(tokens_gold)
                    counts[length] = counts[length] + 1
                    #
                    total_corrects[length] += correct
                    #
                    tokens_correct = 0.0
                    for i in range(ref_min):
                        if tokens_pred[i] == tokens_gold[i]:
                            tokens_correct += 1
                        elif i == 0 or tokens_pred[i-1] == tokens_gold[i-1]:
                            try:
                                gold = vocabulary.index(tokens_gold[i])
                                pred = vocabulary.index(tokens_pred[i])
                                confusion[gold][pred] = confusion[gold][pred] + 1
                            except Exception:
                                print('in exception')
                                code.interact(local=dict(globals(), **locals()))
                    total_tokens_corrects[length] += tokens_correct / ref
                    #
                    edit_distance = distance.levenshtein(tokens_gold, tokens_pred)
                    total_refs[length] += ref
                    total_edit_distances[length] += edit_distance
                    # image evaluation
                    image_edit_distance, image_ref, match1, match2 = \
                        img_edit_distance_file(filename_gold, filename_pred)
                    total_image_edit_distances[length] += image_edit_distance
                    total_image_refs[length] += image_ref
                    if match1:
                        total_image_corrects[length] += 1
                    if match2:
                        total_image_correct_eliminates[length] += 1
                    #
                    correct = 0
                if idx != 0 and idx % 10 == 0:
                    print(idx)
                    #
                    absolute_accuracies = total_corrects / counts
                    token_accuracies = total_tokens_corrects / counts
                    text_edit_distances = 1.0 - total_edit_distances / total_refs
                    image_edit_distances = 1.0 - total_image_edit_distances / total_image_refs
                    image_accuracies = total_image_corrects / counts
                    image_eliminate_accuracies = total_image_correct_eliminates / counts
                    # 
                    token_accuracy = weighted_average(token_accuracies)
                    absolute_accuracy = weighted_average(absolute_accuracies)
                    text_edit_distance = weighted_average(text_edit_distances)
                    image_edit_distance = weighted_average(image_edit_distances)
                    image_accuracy = weighted_average(image_accuracies)
                    image_eliminate_accuracy = weighted_average(image_eliminate_accuracies)
                    #
                    print('token accuracy: ' + str(token_accuracy))
                    print('absolute accuracy: ' + str(absolute_accuracy))
                    print('text edit distance: ' + str(text_edit_distance))
                    print('image_edit_distance: ' + str(image_edit_distance))
                    print('image_accuracy: ' + str(image_accuracy))
                    print('image_eliminate_accuracy: ' + str(image_eliminate_accuracy))
        #
        absolute_accuracies = total_corrects / counts
        token_accuracies = total_tokens_corrects / counts
        text_edit_distances = 1.0 - total_edit_distances / total_refs
        image_edit_distances = 1.0 - total_image_edit_distances / total_image_refs
        image_accuracies = total_image_corrects / counts
        image_eliminate_accuracies = total_image_correct_eliminates / counts
        # 
        token_accuracy = weighted_average(token_accuracies)
        absolute_accuracy = weighted_average(absolute_accuracies)
        text_edit_distance = weighted_average(text_edit_distances)
        image_edit_distance = weighted_average(image_edit_distances)
        image_accuracy = weighted_average(image_accuracies)
        image_eliminate_accuracy = weighted_average(image_eliminate_accuracies)
        #
        print('token accuracy: ' + str(token_accuracy))
        print('absolute accuracy: ' + str(absolute_accuracy))
        print('text edit distance: ' + str(text_edit_distance))
        print('image_edit_distance: ' + str(image_edit_distance))
        print('image_accuracy: ' + str(image_accuracy))
        print('image_eliminate_accuracy: ' + str(image_eliminate_accuracy))
        #
        self.writeResult(self.model_dir + '/' + self.phase + '_token_accuracy', token_accuracy)
        self.writeResult(self.model_dir + '/' + self.phase + '_abs_accuracy', absolute_accuracy)
        self.writeResult(self.model_dir + '/' + self.phase + '_text_edit_distance', \
            text_edit_distance)
        self.writeResult(self.model_dir + '/' + self.phase + '_image_edit_distance', \
            image_edit_distance)
        self.writeResult(self.model_dir + '/' + self.phase + '_image_accuracy', \
            image_accuracy)
        self.writeResult(self.model_dir + '/' + self.phase + '_image_eliminate_accuracy', \
            image_eliminate_accuracy)
        #
        with open(self.evaluation_path + '/analysis.npz', 'w') as fout:
            np.savez(fout, absolute_accuracies=absolute_accuracies, \
                token_accuracies=token_accuracies, \
                text_edit_distances=text_edit_distances, \
                confusion=confusion, \
                total_image_edit_distances=total_image_edit_distances, \
                image_accuracies=image_accuracies, \
                image_eliminate_accuracies=image_eliminate_accuracies)
        # create and save bins
        bins = [0,15,30,45,60,75,90,105,120,135,150,max_length]
        bin_labels = ['<15','[15,30)','[30,45)','[45,60)','[60,75)','[75,90)','[90,105)', \
            '[105,120)','[120,135)', '[135,150)', '>=150']
        def per_bin(metric):
            binc = np.zeros(len(bins) - 1)
            for i in range(0,len(bins)-1):
                binc[i] = weighted_average(metric, bins[i], bins[i+1])
            return binc
        token_accuracy_bins = per_bin(token_accuracies)
        absolute_accuracy_bins = per_bin(absolute_accuracies)
        text_edit_distance_bins = per_bin(text_edit_distances)
        image_edit_distance_bins = per_bin(image_edit_distances)
        image_accuracy_bins = per_bin(image_accuracies)
        image_eliminate_accuracy_bins = per_bin(image_eliminate_accuracies)
        y_pos = np.arange(len(bin_labels))
        def savePlot(metric, name):
            plt.bar(y_pos, metric, align='center', alpha=1.0)
            plt.xticks(y_pos, bin_labels)
            plt.ylabel('%')
            plt.savefig(self.evaluation_path + '/' + name + '.pdf')
            plt.close()
        savePlot(token_accuracy_bins,'token_accuracy_bins')
        savePlot(absolute_accuracy_bins,'absolute_accuracy_bins')
        savePlot(text_edit_distance_bins,'text_edit_distance_bins')
        savePlot(image_edit_distance_bins,'image_edit_distance_bins')
        savePlot(image_accuracy_bins,'image_accuracy_bins')
        savePlot(image_eliminate_accuracy_bins,'image_eliminate_accuracy_bins')
        # extract the most important information of the confusion matrix
        confusion = confusion / np.sum(confusion)
        m = confusion.argsort(axis=None)[::-1][:20]
        m_from = m / len(vocabulary)
        m_to = m % len(vocabulary)
        # code.interact(local=dict(globals(), **locals()))
        for k in range(20):
            vocab_from = vocabulary[m_from[k]]
            vocab_to = vocabulary[m_to[k]]
            value = str(100 * confusion[m_from[k]][m_to[k]])
            line = vocab_from + ' -> ' + vocab_to + ' : ' + value
            self.writeParamInList('confusion_outline.txt', line)
        return token_accuracy, absolute_accuracy, text_edit_distance

    def writeResult(self, write_path, value):
        # write_path = self.evaluation_path + '/' + path
        writer = open(write_path, 'a')
        writer.write(str(value) + '\n')
        writer.close()

    def writeParamInList(self, write_path, value):
        write_path = self.evaluation_path + '/' + write_path
        writer = open(write_path, 'a')
        writer.write(str(value) + '\n')
        writer.close()


def readParamList(read_path):
    if not os.path.exists(read_path):
        print(read_path + ' does not exist!')
        return []
    reader = open(read_path, 'r')
    value = reader.read().split('\n')
    reader.close()
    return value

# return (edit_distance, ref, match, match w/o)
def img_edit_distance(im1, im2, out_path=None):
    img_data1 = np.asarray(im1, dtype=np.uint8) # height, width
    img_data1 = np.transpose(img_data1)
    h1 = img_data1.shape[1]
    w1 = img_data1.shape[0]
    img_data1 = (img_data1<=128).astype(np.uint8)
    if im2:
        img_data2 = np.asarray(im2, dtype=np.uint8) # height, width
        img_data2 = np.transpose(img_data2)
        h2 = img_data2.shape[1]
        w2 = img_data2.shape[0]
        img_data2 = (img_data2<=128).astype(np.uint8)
    else:
        img_data2 = []
        h2 = h1
    if h1 == h2:
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]
    elif h1 > h2:# pad h2
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item])+''.join(['0']*(h1-h2)) for item in img_data2]
    else:
        seq1 = [''.join([str(i) for i in item])+''.join(['0']*(h2-h1)) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]

    seq1_int = [int(item,2) for item in seq1]
    seq2_int = [int(item,2) for item in seq2]
    big = int(''.join(['0' for i in range(max(h1,h2))]),2)
    seq1_eliminate = []
    seq2_eliminate = []
    seq1_new = []
    seq2_new = []
    for idx,items in enumerate(seq1_int):
        if items>big:
            seq1_eliminate.append(items)
            seq1_new.append(seq1[idx])
    for idx,items in enumerate(seq2_int):
        if items>big:
            seq2_eliminate.append(items)
            seq2_new.append(seq2[idx])
    if len(seq2) == 0:
        return (len(seq1), len(seq1), False, False)

    def make_strs(int_ls, int_ls2):
        d = {}
        seen = []
        def build(ls):
            for l in ls:
                if int(l, 2) in d: continue
                found = False
                l_arr = np.array(map(int, l))
            
                for l2,l2_arr in seen:
                    if np.abs(l_arr -l2_arr).sum() < 5:
                        d[int(l, 2)] = d[int(l2, 2)]
                        found = True
                        break
                if not found:
                    d[int(l, 2)] = unichr(len(seen))
                    seen.append((l, np.array(map(int, l))))
                    
        build(int_ls)
        build(int_ls2)
        return "".join([d[int(l, 2)] for l in int_ls]), "".join([d[int(l, 2)] for l in int_ls2])
    #if out_path:
    seq1_t, seq2_t = make_strs(seq1, seq2)

    edit_distance = distance.levenshtein(seq1_int, seq2_int)
    match = True
    if edit_distance>0:
        matcher = StringMatcher(None, seq1_t, seq2_t)

        ls = []
        for op in matcher.get_opcodes():
            if op[0] == "equal" or (op[2]-op[1] < 5):
                ls += [[int(r) for r in l]
                       for l in seq1[op[1]:op[2]]
                       ] 
            elif op[0] == "replace":
                a = seq1[op[1]:op[2]]
                b = seq2[op[3]:op[4]]
                ls += [[int(r1)*3 + int(r2)*2
                        if int(r1) != int(r2) else int(r1)
                        for r1, r2 in zip(a[i] if i < len(a) else [0]*1000,
                                          b[i] if i < len(b) else [0]*1000)]
                       for i in range(max(len(a), len(b)))]
                match = False
            elif op[0] == "insert":

                ls += [[int(r)*3 for r in l]
                       for l in seq2[op[3]:op[4]]]
                match = False
            elif op[0] == "delete":
                match = False
                ls += [[int(r)*2 for r in l] for l in seq1[op[1]:op[2]]]

        #vmax = 3
        #plt.imshow(np.array(ls).transpose(), vmax=vmax)

        #cmap = LinearSegmentedColormap.from_list('mycmap', [(0. /vmax, 'white'),
        #                                                    (1. /vmax, 'grey'),
        #                                                    (2. /vmax, 'blue'),
        #                                                    (3. /vmax, 'red')])

        #plt.set_cmap(cmap)
        #plt.axis('off')
        #plt.savefig(out_path, bbox_inches="tight")

    match1 = match
    seq1_t, seq2_t = make_strs(seq1_new, seq2_new)

    if len(seq2_new) == 0 or len(seq1_new) == 0:
        if len(seq2_new) == len(seq1_new):
            return (edit_distance, max(len(seq1_int),len(seq2_int)), match1, True)# all blank
        return (edit_distance, max(len(seq1_int),len(seq2_int)), match1, False)
    match = True
    matcher = StringMatcher(None, seq1_t, seq2_t)

    ls = []
    for op in matcher.get_opcodes():
        if op[0] == "equal" or (op[2]-op[1] < 5):
            ls += [[int(r) for r in l]
                   for l in seq1[op[1]:op[2]]
                   ] 
        elif op[0] == "replace":
            a = seq1[op[1]:op[2]]
            b = seq2[op[3]:op[4]]
            ls += [[int(r1)*3 + int(r2)*2
                    if int(r1) != int(r2) else int(r1)
                    for r1, r2 in zip(a[i] if i < len(a) else [0]*1000,
                                      b[i] if i < len(b) else [0]*1000)]
                   for i in range(max(len(a), len(b)))]
            match = False
        elif op[0] == "insert":

            ls += [[int(r)*3 for r in l]
                   for l in seq2[op[3]:op[4]]]
            match = False
        elif op[0] == "delete":
            match = False
            ls += [[int(r)*2 for r in l] for l in seq1[op[1]:op[2]]]

    match2 = match

    return (edit_distance, max(len(seq1_int),len(seq2_int)), match1, match2)

def img_edit_distance_file(file1, file2, output_path=None):
    img1 = Image.open(file1).convert('L')
    if os.path.exists(file2):
        img2 = Image.open(file2).convert('L')
    else:
        img2 = None
    return img_edit_distance(img1, img2, output_path)

def main():
    print('enter main method')
    e = Evaluator('/cvhci/data/docs/math_expr/printed/im2latex-100k/mymodels/' + \
        'vggFe_quadroEnc_bahdanauDec_2048_512_momentum_2017-11-08 10:48:07.887550')
    result_path = '/cvhci/data/docs/math_expr/printed/im2latex-100k/mymodels/' + \
        'vggFe_quadroEnc_bahdanauDec_2048_512_momentum_2017-11-08 10:48:07.887550/' + \
        'params/test_results/epoch11'
    vocab_path = '/cvhci/data/docs/math_expr/printed/im2latex-100k/mymodels/' + \
        'vggFe_quadroEnc_bahdanauDec_2048_512_momentum_2017-11-08 10:48:07.887550/' + \
        'params/vocabulary'
    vocabs = readParamList(vocab_path)
    image_dir = '/cvhci/data/docs/math_expr/printed/im2latex-100k/mymodels/' + \
        'vggFe_quadroEnc_bahdanauDec_2048_512_momentum_2017-11-08 10:48:07.887550' + \
        '/params/test_rendered_images'
    e.evaluate(result_path, image_dir, vocabs)
    code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()