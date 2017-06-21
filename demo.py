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

    parser.add_argument('--image-path', dest='image_path',
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
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    parameters = process_args(args)#
    image_path = parameters.image_path
    assert os.path.exists(image_path), image_path
    image = np.array(Image.open(image_path))
    #code.interact(local=locals())
    image = np.transpose(np.array(image),[2,0,1])
    image = image[0]
    imgs = np.expand_dims(np.expand_dims(image,0),3)
    #
    model_path = parameters.model_path
    assert os.path.exists(model_path), model_path
    #
    vocabulary_path = parameters.vocabulary_path
    assert os.path.exists(vocabulary_path), vocabulary_path
    vocabulary = open(vocabulary_path).readlines()
    vocabulary[-1] = vocabulary[-1] + '\n'
    vocabulary = [a[:-1] for a in vocabulary]
    #
    with tf.Graph().as_default():
        sess = tf.Session()
        model = models.wysiwyg.load(model_path, sess)
        feed_dict={model.images_placeholder: imgs}
        network = sess.run(model.network, feed_dict=feed_dict)
        print('hi')
        print(len(vocabulary))
        s = ''
        for i in range(network.shape[1]):
            idx = np.argmax(network[0][i]) # dimensionen richtig??
            print(idx)
            if idx >= len(vocabulary): # welchen index hat das end of line symbol?
                s = s + 'END '
            else:
                s = s + vocabulary[idx] + ' '
        print(s[:-1])



if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
