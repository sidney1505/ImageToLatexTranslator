import sys, os, argparse, logging, json, code, shutil
import numpy as np
from PIL import Image

def generate(params):
    print("batches are generated now!")
    # loads the labels  
    label_path = params.label_path
    #code.interact(local=dict(globals(), **locals()))
    assert os.path.exists(label_path), label_path
    formulas = open(label_path).readlines()
    # loads the connection between labels and the images
    data_path = params.data_path
    assert os.path.exists(data_path), data_path
    # loads the images
    image_dir = params.image_dir
    assert os.path.exists(image_dir), image_dir
    # loads the vocabulary
    vocabulary_path = params.vocabulary_path
    assert os.path.exists(vocabulary_path), vocabulary_path
    vocabulary = open(vocabulary_path).readlines()
    vocabulary[-1] = vocabulary[-1] + '\n'
    vocabulary = [a[:-1] for a in vocabulary]
    # cleans and creates directory to save the batches
    output_path = params.output_path
    shutil.rmtree(output_path, ignore_errors=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # local variables
    buckets = []
    i = 0
    max_num_tokens = 0
    num_classes = 0
    with open(data_path) as fin:
        for line in fin:
            #print('hi')
            image_name, line_idx = line.strip().split()
            line_strip = formulas[int(line_idx)].strip()
            tokens = line_strip.split()
            if len(tokens) > max_num_tokens:
                max_num_tokens = len(tokens)
    with open(data_path) as fin:
        x = 0
        for line in fin:
            image_name, line_idx = line.strip().split()
            line_strip = formulas[int(line_idx)].strip()
            tokens = line_strip.split()
            num_classes = len(vocabulary)+1
            label = np.full(max_num_tokens,num_classes-1) # !!!
            #code.interact(local=dict(globals(), **locals()))
            contained_classes = np.zeros(num_classes)
            k = 0
            for token in tokens:
                if token in vocabulary:
                    label[k] = vocabulary.index(token)
                    contained_classes[label[k]] = 1
                k = k + 1
            image = Image.open(image_dir + '/' + image_name)
            pixr = np.array(image) # rgb wanted???
            h,w,_ = pixr.shape
            pix = np.zeros((h,w,1))
            for n in range(0,h):
                for j in range(0,w):
                    pix[n][j][0] = pixr[n][j][0] # reduce to one channel
            pix = (pix - 128) / 128 # normalization
            j = 0
            b = len(buckets)
            for bucket in buckets:
                if pix.shape == bucket[1][0].shape: # buckets can't be empty                
                    b = j
                    break
                j = j + 1
            if b != len(buckets): # was tuen falls doch???
                name, pixs, labels, contained_classes_list, image_names = buckets[b]
                pixs.append(pix)
                labels.append(label)
                contained_classes_list.append(np.array(contained_classes))
                image_names.append(image_name)
                if len(pixs) >= 1000: # 1000 optimale wahl???
                    with open(output_path + "/" + name, 'w') as fout:
                        np.savez(fout, images=np.array(pixs), labels=np.array(labels), \
                            num_classes=num_classes, \
                            contained_classes_list=contained_classes_list, \
                            image_names=image_names)
                        print(name + ' saved! ' + str(np.array(pixs).shape) + ' : ' + \
                            str(np.array(labels).shape))
                    buckets.pop(b)
                else:
                    buckets[b] = (name, pixs, labels, contained_classes_list, image_names)
            else:
                buckets.append(('batch' + str(i) + '.npz', [pix], [label], \
                    [contained_classes], [image_name]))
                i = i + 1
    for batch in buckets:
        name, pixs, labels, contained_classes_list, image_names = batch 
        with open(output_path + "/" + name, 'w') as fout:
            np.savez(fout, images=np.array(pixs), labels=np.array(labels), \
                num_classes=num_classes, contained_classes_list=contained_classes_list, \
                image_names=image_names)
            print(name + ' saved! ' + str(np.array(pixs).shape) + ' : ' + \
                str(np.array(labels).shape))
    print(num_classes)

def process_args(args):
    parser = argparse.ArgumentParser(description='description')

    parser.add_argument('--image-dir', dest='image_dir',
                        type=str, default=os.environ['IMAGE_DIR'],
                        help=('Directory containing processed images.'
                        ))
    parser.add_argument('--data-path', dest='data_path',
                        type=str, default=os.environ['TRAIN_FILTER'],
                        help=('Input file path containing <label_idx> <img_path> \
                            <mode> per line. Note that <img_path> does not contain postfix.'
                        ))
    parser.add_argument('--output-path', dest='output_path',
                        type=str, default=os.environ['TBD'],
                        help=('Path of the saved batches'))
    parser.add_argument('--label-path', dest='label_path',
                        type=str, default=os.environ['FORMULAS'],
                        help=('Input label path containing <formula> per line. This is \
                            required if filter flag is set, and data point with blank \
                            formulas will be discarded.'
                        ))
    parser.add_argument('--vocabulary-path', dest='vocabulary_path',
                        type=str, default=os.environ['VOCABS'],
                        help=('List of existing tokens'))
    params = parser.parse_args(args)
    return params

def main(args):
    params = process_args(args)
    generate(params)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
