import sys, os, argparse, logging, json
import numpy as np
import PIL
from PIL import Image

# np.savez('bla.npz', image=image, label=label)


# data = np.load('bla.npz')
# image = data['image']

def process_args(args):
    parser = argparse.ArgumentParser(description='description')

    parser.add_argument('--image-dir', dest='image_dir',
                        type=str, default='',
                        help=('Directory containing processed images.'
                        ))
    parser.add_argument('--data-path', dest='data_path',
                        type=str, required=True,
                        help=('Input file path containing <label_idx> <img_path> <mode> per line. Note that <img_path> does not contain postfix.'
                        ))
    parser.add_argument('--output-path', dest='output_path',
                        type=str, required=True,
                        help=('Path of the saved batches'))
    parser.add_argument('--label-path', dest='label_path',
                        type=str, default='',
                        help=('Input label path containing <formula> per line. This is required if filter flag is set, and data point with blank formulas will be discarded.'
                        ))
    #parser.add_argument('--buckets', dest='buckets',
    #                    type=str, default='[[240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100], [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100], [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200], [1000, 400], [1200, 200], [1600, 200], [1600, 1600]]',
    #                    help=('Bucket sizes used for grouping. Should be a Json string. Note that this denotes the bucket size after padding and before downsampling.'
    #                    ))
    parser.add_argument('--vocabulary-path', dest='vocabulary_path',
                        type=str, required=True,
                        help=('List of existing tokens'))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    parameters = process_args(args)
    label_path = parameters.label_path
    assert os.path.exists(label_path), label_path
    formulas = open(label_path).readlines()
    data_path = parameters.data_path
    assert os.path.exists(data_path), data_path
    # funktioniert auch mit directories???
    image_dir = parameters.image_dir
    assert os.path.exists(image_dir), image_dir
    vocabulary_path = parameters.vocabulary_path
    assert os.path.exists(vocabulary_path), vocabulary_path
    vocabulary = open(vocabulary_path).readlines()
    vocabulary[-1] = vocabulary[-1] + '\n'
    vocabulary = [a[:-1] for a in vocabulary]
    # clean directory
    output_path = parameters.output_path
    # buckets = json.loads(parameters.buckets)
    buckets = []
    # batch_groups = []
    i = 0
    #for bucket in buckets:
    #	batch_groups.append(('batch' + str(i) + '.npz', [], []))
    #	i = i + 1
    with open(data_path) as fin:
        for line in fin:
            #print('buckets')
            #print(len(buckets))
            #print('i')
            #print(i)
            image_name, line_idx = line.strip().split()
            line_strip = formulas[int(line_idx)].strip()
            tokens = line_strip.split()
            label = np.full(150,len(vocabulary)) # 150 richtige wahl???
            k = 0
            for token in tokens:
            	if token in vocabulary: # was sonst???
            		label[k] = vocabulary.index(token)
            	k = k + 1
            # wie encode ich die label?? one-hot?? int??
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
            # h, w = pix.shape # rgb pictures
            # print('(' + str(h) + ',' + str(w) + ')')
            for bucket in buckets:            	
            	# print(type(bucket))
            	# print("(" + str(bucket[0]) + "," + str(bucket[1]) + ")")
                # import code
                #code.interact(local=locals())
                #print(pix.shape)
                #print(bucket[1][0].shape)
            	if pix.shape == bucket[1][0].shape: # buckets can't be empty           		
            		b = j
            		break
            	j = j + 1
            if b != len(buckets): # was tuen falls doch???
                #print('OK')
            	name, pixs, labels = buckets[b]
            	pixs.append(pix)
            	labels.append(label)
            	if len(pixs) >= 1000: # 1000 optimale wahl???
            		with open(output_path + "/" + name, 'w') as fout:
    					np.savez(fout, images=np.array(pixs), labels=np.array(labels))
                        print(name + ' saved!')
            		buckets.pop(b)
            	else:
            		buckets[b] = (name, pixs, labels)
            else:
                buckets.append(('batch' + str(i) + '.npz', [pix], [label]))
                i = i + 1
    for batch in buckets:
    	name, pixs, labels = batch 
    	with open(output_path + "/" + name, 'w') as fout:
            np.savez(fout, images=np.array(pixs), labels=np.array(labels))
            print(name + ' saved!')

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
