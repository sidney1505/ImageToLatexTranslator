import sys, os, argparse, logging, json
import shutil
import numpy as np
import tensorflow as tf
import PIL
import code
from PIL import Image
from models.wysiwyg import Model
import models.wysiwyg

# np.savez('bla.npz', image=image, label=label)


# data = np.load('bla.npz')
# image = data['image']

def process_args(args):
    parser = argparse.ArgumentParser(description='description')

    parser.add_argument('--batch-path', dest='batch_path',
                        type=str, default='',
                        help=('Directory containing processed images.'
                        ))
    parser.add_argument('--index', dest='index',
                        type=str, default='',
                        help=('Directory containing processed images.'
                        ))
    parser.add_argument('--model-path', dest='model_path',
                        type=str, default='',
                        help=('Directory containing model.'
                        ))
    parser.add_argument('--vocabulary-path', dest='vocabulary_path',
                        type=str, required=True,
                        help=('List of existing tokens'))
    parser.add_argument('--train-type', dest='train_type',
                        type=str, default=5,
                        help=('List of existing tokens'))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    params = process_args(args)#
    params.index = int(params.index)
    params.train_type = int(params.train_type)
    assert os.path.exists(params.batch_path), params.batch_path
    assert os.path.exists(params.vocabulary_path), params.vocabulary_path
    assert os.path.exists(params.model_path), params.model_path
    batch = np.load(params.batch_path)
    num_classes = batch['num_classes']
    max_num_tokens = len(batch['labels'][0])
    #
    image = np.array([batch['images'][params.index]])
    label = batch['labels'][params.index]
    vocabulary = open(params.vocabulary_path).readlines()
    vocabulary[-1] = vocabulary[-1] + '\n'
    vocabulary = [a[:-1] for a in vocabulary]
    #
    with tf.Graph().as_default():
        code.interact(local=dict(globals(), **locals()))
        model = models.wysiwyg.load(params.model_path, params.train_type)
        prediction = model.predict(image)
        label_prediction = ''
        label_gold = ''
        for i in range(max_num_tokens):
            idx = np.argmax(prediction[0][i])
            if idx >= len(vocabulary):
                label_prediction = label_prediction + 'END '
            else:
                label_prediction = label_prediction + vocabulary[idx] + ' '
            idx = label[i]
            if idx >= len(vocabulary):
                label_prediction = label_prediction + 'END '
            else:
                label_prediction = label_prediction + vocabulary[idx] + ' '
        print('Prediction:')
        print(label_prediction[:-1])
        print('Groundtruth:')
        print(label_gold[:-1])



if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
