import sys, os, argparse, logging, json
import shutil
import numpy as np
import PIL
from PIL import Image

# np.savez('bla.npz', image=image, label=label)


# data = np.load('bla.npz')
# image = data['image']

def process_args(args):
    parser = argparse.ArgumentParser(description='description')

    parser.add_argument('--image-path', dest='image_path',
                        type=str, default='',
                        help=('Directory containing processed images.'
                        ))
    parser.add_argument('--weights-path', dest='weights_path',
                        type=str, default='',
                        help=('Directory containing weights.'
                        ))
    parser.add_argument('--vocabulary-path', dest='vocabulary_path',
                        type=str, required=True,
                        help=('List of existing tokens'))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    parameters = process_args(args)#
    image_path = parameters.image_path
    assert os.path.exists(image_path), image_path
    image = open(image_path)
    weights_path = parameters.weights_path
    assert os.path.exists(weights_path), weights_path
    vocabulary_path = parameters.vocabulary_path
    assert os.path.exists(vocabulary_path), vocabulary_path
    vocabulary = open(vocabulary_path).readlines()
    vocabulary[-1] = vocabulary[-1] + '\n'
    vocabulary = [a[:-1] for a in vocabulary]
    # woher die paramter???
    model = Model(num_classes, max_num_tokens, 0.01)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
