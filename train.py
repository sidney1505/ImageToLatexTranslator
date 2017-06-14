import sys, os, argparse, logging
import numpy as np
import code
# from model.model import Model
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from models.wysiwyg import Model

def process_args(args):
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--phase', dest='phase',
                        type=str, default='prediction',
                        help=('Directory containing processed images.'))
    parser.add_argument('--batch-dir', dest='batch_dir',
                        type=str, required=True,
                        help=('path where the batches are stored'))
    parser.add_argument('--batch-size', dest='batch_size',
                        type=str, default=2,
                        help=('size of the minibatches'))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    parameters = process_args(args)
    print("main")
    phase = parameters.phase
    batch_dir = parameters.batch_dir
    assert os.path.exists(batch_dir), batch_dir
    batchfiles = os.listdir(batch_dir)
    batchsize = parameters.batch_size
    batch0 = np.load(batch_dir + '/batch0.npz')
    num_classes = batch0['num_classes']
    max_num_tokens = len(batch0['labels'][0])
    with tf.Graph().as_default():
        model = Model(num_classes, max_num_tokens, batchsize)
        sess = tf.Session()
        labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None,max_num_tokens])
        # Add to the Graph the Ops for loss calculation.
        loss = model.loss(model.network, labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = model.training(loss)
        # Add the variable initializer Op.
        init = tf.global_variables_initializer()
        sess.run(init)
        # broken = [(0,46),(0,47),(1,68),(1,69)]
        for i in range(model.nr_epochs):
            k = 0
            for batchfile in batchfiles: # randomise!!!
                if k < 0:
                    k = k + 1
                    continue             
                print('load ' + batchfile + '!')
                batch = np.load(batch_dir + '/' + batchfile)
                images = batch['images']
                labels = batch['labels']
                assert len(images) == len(labels) != 0                
                for j in range(len(images) / model.batchsize):
                    if j > len(images) / model.batchsize - 2:
                        break
                    print('batch(' + str(k) + ',' + str(j*model.batchsize) + '):')
                    print(images.shape)
                    imgs = []
                    labs = []
                    for b in range(j*model.batchsize, min((j+1)*model.batchsize,len(images))):
                        imgs.append(images[b])
                        labs.append(labels[b])
                    imgs = np.array(imgs)
                    labs = np.array(labs)
                    # code.interact(local=locals())
                    feed_dict={model.images_placeholder: imgs, labels_placeholder: labs}
                    if phase == "training":
                        _, loss_value, network = sess.run([train_op, loss, model.network], feed_dict=feed_dict)
                        print(network.shape)
                        print(loss_value)
                    elif phase == "prediction":
                        network = sess.run(model.network, feed_dict=feed_dict)
                        print(network.shape)
                k = k + 1
        sess.close()        

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')