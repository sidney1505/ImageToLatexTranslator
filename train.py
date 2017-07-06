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
import models.wysiwyg

def process_args(args):
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--phase', dest='phase',
                        type=str, default='training',
                        help=('The phase (training or prediction).'))
    parser.add_argument('--traintype', dest='traintype',
                        type=str, default=1,
                        help=('Type of the training/evaluation.\
                             0 = normal, 1 = pretraining cnn only, \
                             2 = encoder refinement only, \
                             3 = decoder only'))
    parser.add_argument('--load-model', dest='load_model',
                        type=str, default=False,
                        help=('size of the minibatches'))
    parser.add_argument('--train-batch-dir', dest='train_batch_dir',
                        type=str, required=True,
                        help=('Path where the training \
                            batches are stored'))
    parser.add_argument('--val-batch-dir', dest='val_batch_dir',
                        type=str, required=True,
                        help=('Path where the validation \
                            batches are stored'))
    parser.add_argument('--model-dir', dest='model_dir',
                        type=str, required=True,
                        help=('Path where the model is stored \
                            if it already exists or \
                            should be stored if not.'))
    parser.add_argument('--capacity', dest='capacity',
                        type=str, default=100000,
                        help=('The total amount of floats \
                            that can be propagated through \
                            the network at a time. \
                            Influences the batch size!'))
    parameters = parser.parse_args(args)
    return parameters

def calculateMinibatchsize(image, capacity):
    minibatchsize = 20
    while image.shape[0] * image.shape[1] * image.shape[2] * minibatchsize > capacity:
        minibatchsize = minibatchsize - 1
    return minibatchsize

def loadBatch(batch_dir, batch_names, batch_it, traintype, capacity):
    batch = None
    batch_images = None
    new_iteration = False
    minibatchsize = None
    batch_labels = None
    #code.interact(local=dict(globals(), **locals()))
    while True:
        batch = np.load(batch_dir + '/' + batch_names[batch_it])
        batch_images = batch['images']
        #code.interact(local=dict(globals(), **locals()))
        minibatchsize = calculateMinibatchsize(batch_images[0], \
            capacity)
        if minibatchsize != 0:
            break
        else:
            new_iteration = (batch_it + 1 >= len(batch_names))
            batch_it = (batch_it + 1) % len(batch_names)
            assert batch_it != 0, 'capacity to small for training data'
    if traintype in [0,3,4]:
        batch_labels = batch['labels']
    elif traintype in [1,2]:
        batch_labels = batch['contained_classes_list']
    classes_true = batch['contained_classes_list']
    new_iteration = (batch_it + 1 >= len(batch_names))
    batch_it = (batch_it + 1) % len(batch_names)
    assert len(batch_images) == len(batch_labels) != 0
    return batch_images, minibatchsize, batch_it, batch_labels, new_iteration, \
        classes_true

def createMinibatch(batch_data, minibatch_it, minibatchsize):
    minibatch_data = []
    for it in range(minibatch_it * minibatchsize, min((minibatch_it + 1) * \
            minibatchsize,len(batch_data))):
        minibatch_data.append(batch_data[it])
    return np.array(minibatch_data)

def calculateAccuracy(classes_pred, classes_true, stats):
    accuracy = 0.0
    for ccpb in range(classes_pred.shape[0]):
        for ccp in range(classes_pred.shape[1]):
            if classes_pred[ccpb][ccp] >= 0.5 and classes_true[ccpb][ccp] == 1 \
                    or classes_pred[ccpb][ccp] < 0.5 and classes_true[ccpb][ccp] == 0:
                accuracy = accuracy + 1.0
                stats[ccp] = stats[ccp] + 1
        stats[-1] = stats[-1] + 1
    return accuracy / (classes_pred.shape[0] * classes_pred.shape[1]), stats

def main(args):
    params = process_args(args)
    params.traintype = int(params.traintype)
    params.capacity = int(params.capacity)
    params.load_model = bool(params.load_model)
    # check if given paths really exist
    assert os.path.exists(params.train_batch_dir), \
        "Train batch directory doesn't exists!"
    assert os.path.exists(params.val_batch_dir), \
        "Validation batch directory doesn't exists!"
    assert os.path.exists(params.model_dir) or not params.load_model, \
        "Model directory doesn't exists!"

    # load the names of the training & validation batchfiles
    train_batch_names = os.listdir(params.train_batch_dir)
    assert train_batch_names != [], \
        "Training batch directory is empty!"
    val_batch_names = os.listdir(params.val_batch_dir)
    assert val_batch_names != [], \
        "Validation batch directory is empty!"

    # extract some needed paramters from the first training batch
    batch0 = np.load(params.train_batch_dir + '/' + train_batch_names[0])
    num_classes = batch0['num_classes']
    max_num_tokens = len(batch0['labels'][0])    

    # creates the tensorflow session were the tensorflow variables can live
    with tf.Graph().as_default():
        sess = tf.Session()

        # load/create the model
        model = None
        if params.load_model:
            model = models.wysiwyg.load(params.model_dir, sess)
            #code.interact(local=dict(globals(), **locals()))
            # intialise uninitialised variables???
        else:
            # create the model directory
            if not os.path.exists(params.model_dir):
                os.makedirs(params.model_dir)
            params.model_dir = params.model_dir + '/model-' + \
                str(datetime.datetime.now())
            if not os.path.exists(params.model_dir):
                os.makedirs(params.model_dir)
            model = Model(num_classes, max_num_tokens, params.model_dir, capacity= \
                params.capacity, train_mode=params.traintype)       

        # define used groundtruth, used loss function and used train_op
        labels_placeholder = None
        loss = None
        classes = None
        images_placeholder = None
        train_op = tf.constant(0)
        if params.traintype == 0:
            labels_placeholder = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])
            loss = model.loss(model.network, labels_placeholder)
            classes = model.classes
            images_placeholder = model.images_placeholder
        elif params.traintype == 1:
            labels_placeholder = tf.placeholder(dtype=tf.float32, \
                shape=[None,num_classes])
            loss = model.containedClassesLoss(model.containedClassesPrediction, \
                labels_placeholder)
            classes = model.classes
            images_placeholder = model.images_placeholder
        elif params.traintype == 2:
            labels_placeholder = tf.placeholder(dtype=tf.float32, \
                shape=[None,num_classes])
            loss = model.containedClassesLoss( \
                model.containedClassesPredictionRefined, labels_placeholder)
            classes = model.classesRefined
            images_placeholder = model.images_placeholder2
        elif params.traintype == 3:
            labels_placeholder = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])
            loss = model.loss(model.prediction, labels_placeholder)
            classes = tf.constant(0)
            images_placeholder = model.images_placeholder2
        elif params.traintype == 4:
            labels_placeholder = tf.placeholder(dtype=tf.int32, \
                shape=[None,max_num_tokens])
            loss = model.loss(model.prediction, labels_placeholder)
            classes = tf.constant(0)
            images_placeholder = model.images_placeholder
        if params.phase == 'training':
            train_op = model.training(loss)
        # initialises all variables
        if not params.load_model:
            init = tf.global_variables_initializer()
            sess.run(init)
        # intial save! save seeds??
        model.save(sess)
        
        # creates the first validation batch
        val_batch_it = 0
        #code.interact(local=dict(globals(), **locals()))
        val_batch_images, val_minibatchsize, val_batch_it, val_batch_labels,_, \
            val_classes_true = loadBatch(params.val_batch_dir, val_batch_names, \
            val_batch_it, params.traintype, model.capacity)
        # validation minibatch iterator needs to be created
        val_minibatch_it = 0
        val_stats = np.zeros(model.num_classes + 1)
        # the training loop
        for epoch in range(model.nr_epochs):
            train_batch_it = 0
            train_batch_images, train_minibatchsize, train_batch_it, \
                train_batch_labels, new_train_iteration, train_batch_classes_true = \
                loadBatch(params.train_batch_dir, train_batch_names, train_batch_it, \
                params.traintype, model.capacity)
            val_minibatch_loss_old = float("inf")
            train_stats = np.zeros(model.num_classes + 1)
            while not new_train_iteration: # randomise!!!
                print(train_batch_it)
                train_batch_losses = []
                train_batch_accuracies = []
                for train_minibatch_it in range(train_batch_images.shape[0] \
                        / train_minibatchsize):
                    # create the minibatches
                    train_minibatch_images = createMinibatch(train_batch_images, \
                        train_minibatch_it, train_minibatchsize)
                    train_minibatch_labels = createMinibatch(train_batch_labels, \
                        train_minibatch_it, train_minibatchsize)
                    train_minibatch_classes_true = createMinibatch( \
                        train_batch_classes_true, \
                        train_minibatch_it, train_minibatchsize)
                    # create the dict, that is fed into the placeholders
                    #code.interact(local=dict(globals(), **locals()))
                    feed_dict={images_placeholder: train_minibatch_images, \
                        labels_placeholder: train_minibatch_labels}
                    #code.interact(local=dict(globals(), **locals()))
                    print(train_minibatch_images.shape)
                    _, train_minibatch_loss_value, train_minibatch_classes_pred \
                        = sess.run([train_op, loss, classes], \
                        feed_dict=feed_dict)
                    
                    if model.train_mode >= 3:
                        train_minibatch_accuracy = 1.0
                    else:
                        train_minibatch_accuracy, train_stats = calculateAccuracy( \
                            train_minibatch_classes_pred, train_minibatch_classes_true, \
                            train_stats)

                    print('minibatch(' + str(epoch) + ',' + str(train_batch_it) \
                        + ',' + str(train_minibatch_it*train_minibatchsize)+') : '+\
                        str(train_minibatch_images.shape) + ' : ' + \
                        str(train_minibatch_labels.shape) + ' : ' + \
                        str(train_minibatch_loss_value) + ' : ' + \
                        str(train_minibatch_accuracy))
                    train_batch_losses.append(train_minibatch_loss_value)
                    train_batch_accuracies.append(train_minibatch_accuracy)
                # calculating the validation loss
                # create the minibatches
                val_minibatch_images = createMinibatch(val_batch_images, \
                    val_minibatch_it, val_minibatchsize)
                val_minibatch_labels = createMinibatch(val_batch_labels, \
                    val_minibatch_it, val_minibatchsize)
                val_minibatch_classes_true = createMinibatch( \
                    val_classes_true, val_minibatch_it, val_minibatchsize)
                # create the dict, that is fed into the placeholders
                feed_dict={images_placeholder: val_minibatch_images, \
                    labels_placeholder: val_minibatch_labels}
                val_minibatch_loss_value, val_minibatch_classes_pred \
                    = sess.run([loss, classes], \
                    feed_dict=feed_dict)
                
                if model.train_mode >= 3:
                    val_minibatch_accuracy = 1.0
                else:
                    val_minibatch_accuracy, val_stats = calculateAccuracy( \
                        val_minibatch_classes_pred, val_minibatch_classes_true, val_stats)
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')            
                print('minibatch(' + str(epoch) + ',' + str(val_batch_it) \
                    + ',' + str(val_minibatch_it * val_minibatchsize) + ') : ' + \
                    str(val_minibatch_images.shape) + ' : ' + \
                    str(val_minibatch_labels.shape) + ' : ' + \
                    str(val_minibatch_loss_value) + ' : ' + \
                    str(val_minibatch_accuracy))
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                # adds a log
                model.addLog(np.mean(train_batch_losses), val_minibatch_loss_value, \
                    np.mean(train_batch_accuracies), val_minibatch_accuracy, epoch, \
                    train_batch_it, train_stats, val_stats)
                val_minibatch_it = val_minibatch_it + 1
                if (val_minibatch_it + 1) * val_minibatchsize > len(val_batch_images):
                    val_batch_images, val_minibatchsize, val_batch_it, \
                        val_batch_labels,_,val_classes_true = \
                        loadBatch(params.val_batch_dir, \
                        val_batch_names, val_batch_it, params.traintype,\
                        model.capacity)
                    val_minibatch_it = 0                
                train_batch_images, train_minibatchsize, train_batch_it, \
                train_batch_labels, new_train_iteration, train_batch_classes_true = \
                    loadBatch(params.train_batch_dir, train_batch_names, \
                    train_batch_it, params.traintype, model.capacity)
                if val_minibatch_loss_old < val_minibatch_loss_value:
                    print('decrease learning rate!')
                    model.learning_rate = model.learning_rate * 0.5
                val_minibatch_loss_old = val_minibatch_loss_value
                # saves the model
                model.save(sess)
        sess.close()

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')