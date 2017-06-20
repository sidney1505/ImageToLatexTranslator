import sys, os, argparse, logging
import numpy as np
import code
import shutil
import datetime
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
                        type=str, default='training',
                        help=('Directory containing processed images.'))
    parser.add_argument('--batch-dir', dest='batch_dir',
                        type=str, required=True,
                        help=('path where the batches are stored'))
    parser.add_argument('--val-dir', dest='val_dir',
                        type=str, required=True,
                        help=('path where the validation batches are stored'))
    parser.add_argument('--store-dir', dest='store_dir',
                        type=str, default='SavedModels',
                        help=('path where the models are stored'))
    parser.add_argument('--batch-size', dest='batch_size',
                        type=str, default=2,
                        help=('size of the minibatches'))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    parameters = process_args(args)
    # the phase, training or prediction, that is performed
    phase = parameters.phase
    # the directory, where the training batches are stored
    batch_dir = parameters.batch_dir
    assert os.path.exists(batch_dir), batch_dir
    # the training batchfiles
    batchfiles = os.listdir(batch_dir)
    batchsize = parameters.batch_size
    batch0 = np.load(batch_dir + '/batch0.npz')
    # extract some needed paramters from the first batch which has to exist
    num_classes = batch0['num_classes']
    max_num_tokens = len(batch0['labels'][0])
    # the directory where the validation batches are stored
    val_dir = parameters.val_dir
    assert os.path.exists(val_dir), val_dir
    validation_batches = os.listdir(val_dir)
    validation_batches_it = 0
    val_minibatchsize = 1
    while True:
        val_batch = np.load(batch_dir + '/' + validation_batches[validation_batches_it])
        val_images = val_batch['images']
        val_k = 20
        while val_images.shape[1] * val_images.shape[2] * val_k > 50000:
            val_k = val_k - 1
        val_minibatchsize = val_k
        if val_k != 0:
            break
        else:
            validation_batches_it = validation_batches_it + 1 % len(validation_batches)
    val_labels = val_batch['labels']
    val_j = 0
    val_imgs = []
    val_labs = []
    val_losses = []
    # the directory where the models are stored
    store_dir = parameters.store_dir
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    model_dir = store_dir + '/model-' + str(datetime.datetime.now())
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with tf.Graph().as_default():
        sess = tf.Session()
        model = Model(num_classes, max_num_tokens, 0.01,model_dir=model_dir)
        labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None,max_num_tokens])
        # Add to the Graph the Ops for loss calculation.
        loss = model.loss(model.network, labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = model.training(loss)
        # Add the variable initializer Op.
        init = tf.global_variables_initializer()
        sess.run(init)
        model.save(sess)
        print('count of variables:')
        count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(count)
        for i in range(model.nr_epochs):
            val_loss_list = []
            k = 0
            for batchfile in batchfiles: # randomise!!!
                print('load ' + batchfile + '!')
                batch = np.load(batch_dir + '/' + batchfile)
                images = batch['images']
                train_k = 20
                while images.shape[1] * images.shape[2] * train_k > 50000:
                    train_k = train_k - 1
                if train_k == 0:
                    continue
                model.minibatchsize = train_k
                labels = batch['labels']
                print(images.shape)
                assert len(images) == len(labels) != 0
                losses = []
                for j in range(len(images) / model.minibatchsize):
                    #if j > len(images) / model.batchsize - 2:
                    #    break
                    print('batch(' + str(k) + ',' + str(j*model.minibatchsize) + '):')
                    print(images.shape)
                    imgs = []
                    labs = []
                    for b in range(j*model.minibatchsize, min((j+1)*model.minibatchsize,len(images))):
                        imgs.append(images[b])
                        labs.append(labels[b])
                    imgs = np.array(imgs)
                    labs = np.array(labs)
                    # code.interact(local=locals())
                    feed_dict={model.images_placeholder: imgs, labels_placeholder: labs}
                    if phase == "training":
                        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                        print(loss_value)
                        losses.append(loss_value)
                        if j == len(images) / model.minibatchsize - 1:
                            model.addTrainLog(np.mean(loss_value), i, k)
                            # calculating the validation loss
                            val_imgs = []
                            val_labs = []
                            for b in range(val_j*val_minibatchsize, min((val_j+1)*val_minibatchsize,len(val_images))):
                                val_imgs.append(val_images[b])
                                val_labs.append(val_labels[b])
                            val_imgs = np.array(val_imgs)
                            val_labs = np.array(val_labs)
                            val_feed_dict={model.images_placeholder: val_imgs, labels_placeholder: val_labs}
                            val_loss_value = sess.run(loss, feed_dict=val_feed_dict)
                            val_loss_list.append(val_loss_value)
                            model.addValidationLog(val_loss_value, i, k)
                            val_j = val_j + 1
                            if (val_j + 1) * val_minibatchsize > len(val_images):
                                val_j = 0
                                if validation_batches_it == len(validation_batches) - 1:
                                    validation_batches_it = 0
                                else:
                                    validation_batches_it = validation_batches_it + 1
                                while True:
                                    val_batch = np.load(batch_dir + '/' + validation_batches[validation_batches_it])
                                    val_images = val_batch['images']
                                    val_k = 20
                                    while val_images.shape[1] * val_images.shape[2] * val_k > 50000:
                                        val_k = val_k - 1
                                    val_minibatchsize = val_k
                                    if val_k != 0:
                                        break
                                    else:
                                        validation_batches_it = validation_batches_it + 1 % len(validation_batches)
                                val_labels = batch['labels']                                
                    elif phase == "prediction":
                        network = sess.run(model.network, feed_dict=feed_dict)
                        print(network.shape)
                k = k + 1
            val_losses.append(np.mean(val_loss_list))
            if len(val_losses) > 1 and val_losses[-2] <= val_losses[-1]:
                print('decrease learning rate!')
                model.learning_rate = model.learning_rate * 0.5
            model.save(sess)
        sess.close()        

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')