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
    parser.add_argument('--model-path', dest='model_path',
                        type=str, default='',
                        help=('Directory containing model.'
                        ))
    parser.add_argument('--batch-dir', dest='batch_dir',
                        type=str, required=True,
                        help=('path where the batches are stored'))
    parser.add_argument('--output-dir', dest='output_dir',
                        type=str, required=True,
                        help=('path where the new batches will be stored'))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    parameters = process_args(args)
    # the directory, where the training batches are stored
    batch_dir = parameters.batch_dir
    assert os.path.exists(batch_dir), batch_dir
    batchfiles = os.listdir(batch_dir)
    #
    model_path = parameters.model_path
    assert os.path.exists(model_path), model_path
    #
    output_dir = parameters.output_dir
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    #
    with tf.Graph().as_default():
        sess = tf.Session()
        model = models.wysiwyg.load(model_path, sess)
        for batchfile in batchfiles:
            batch = np.load(batch_dir + '/' + batchfile)
            print(batchfile + ' loaded!')
            preds = []
            images = batch['images']
            minibatchsize = 20
            #code.interact(local=dict(globals(), **locals()))
            while images.shape[1] * images.shape[2] * minibatchsize > model.capacity:
                minibatchsize = minibatchsize - 1
            if minibatchsize == 0:
                print('skip')
                continue
            for j in range(len(images) / minibatchsize + 1):
                imgs = []
                for b in range(j*minibatchsize, min((j+1)* \
                        minibatchsize,len(images))):
                    imgs.append(images[b])
                if imgs == []:
                    continue
                imgs = np.array(imgs)
                #print(str(j) + ' : ' + str(minibatchsize) + ' : ' + str(images.shape[0]) + \
                #    ' : ' + str(imgs.shape))
                feed_dict={model.images_placeholder: imgs}
                pred = sess.run(model.features, \
                    feed_dict=feed_dict)
                preds.append(pred)
            preds = np.concatenate(preds)
            assert preds.shape[0] == images.shape[0] != 0 
            with open(output_dir + "/" + batchfile, 'w') as fout:
                np.savez(fout, images=preds, labels=batch['labels'], \
                    num_classes=batch['num_classes'], \
                    contained_classes_list=batch['contained_classes_list'])
            print(batchfile + ' saved!')



if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
