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
    parser.add_argument('--image-dir', dest='image_dir',
                        type=str, default='',
                        help=('Directory containing processed images.'
                        ))
    parser.add_argument('--image-id', dest='image_id',
                        type=str, default=0,
                        help=('Directory containing processed images.'
                        ))
    parser.add_argument('--data-path', dest='data_path',
                        type=str, default='',
                        help=('Directory containing processed images.'
                        ))
    parser.add_argument('--label-path', dest='label_path',
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
    # loads the connection between labels and the images
    label_path = parameters.label_path
    assert os.path.exists(label_path), label_path
    formulas = open(label_path).readlines()
    # loads the connection between labels and the images
    data_path = parameters.data_path
    assert os.path.exists(data_path), data_path
    # loads the images
    image_dir = parameters.image_dir
    assert os.path.exists(image_dir), image_dir
    # loads the image_id
    image_id = int(parameters.image_id)
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
        model = models.wysiwyg.load(model_path, sess, 1)
        with open(data_path) as fin:
            #code.interact(local=locals())
            lines = fin.readlines()
            line = lines[image_id]
            image_name, line_idx = line.strip().split()
            line_strip = formulas[int(line_idx)].strip()
            tokens = line_strip.split()
            contained_classes = np.zeros(model.num_classes)
            for token in tokens:
                if token in vocabulary:
                    contained_classes[vocabulary.index(token)] = 1
            #
            image_path = image_dir + '/' + image_name
            image = np.array(Image.open(image_path))
            image = np.transpose(np.array(image),[2,0,1])
            image = image[0]
            imgs = np.expand_dims(np.expand_dims(image,0),3)
        feed_dict={model.images_placeholder: imgs}
        pred = sess.run(model.classes, feed_dict=feed_dict)
        pred = np.squeeze(pred)
        preds = {}
        for i in range(len(vocabulary)):
            preds.update({pred[i]:(contained_classes[i],vocabulary[i])})
        for key in sorted(preds):
            ccs, voc =  preds[key]
            print(str(key) + '  :  ' + str(ccs) + '  :  ' + voc)
        code.interact(local=dict(globals(), **locals())) 
    print(image_name)
    print(line_strip)



if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
