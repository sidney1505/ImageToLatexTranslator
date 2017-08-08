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
from models.ModelFactory import Model
import models.ModelFactory

def calculateMinibatchsize(image, capacity):
    minibatchsize = 20
    while image.shape[0] * image.shape[1] * image.shape[2] * minibatchsize > capacity:
        minibatchsize = minibatchsize - 1
    return minibatchsize

def loadBatch(batch_dir, batch_names, batch_it, model, capacity):
    batch = None
    batch_images = None
    new_iteration = False
    minibatchsize = None
    batch_labels = None
    batch_it_old = batch_it
    #code.interact(local=dict(globals(), **locals()))
    while True:
        batch = np.load(batch_dir + '/' + batch_names[batch_it])
        batch_images = batch['images']
        batch_imgnames = batch['image_names']
        #code.interact(local=dict(globals(), **locals()))
        minibatchsize = calculateMinibatchsize(batch_images[0], \
            capacity)
        if minibatchsize != 0:
            break
        else:
            new_iteration = (batch_it + 1 >= len(batch_names))
            batch_it = (batch_it + 1) % len(batch_names)
            assert batch_it != batch_it_old, 'capacity to small for training data'
    if model.used_loss == 'label':
        batch_labels = batch['labels']
    elif model.used_loss == 'classes':
        batch_labels = batch['contained_classes_list']
    classes_true = batch['contained_classes_list']
    new_iteration = (batch_it + 1 >= len(batch_names))
    batch_it = (batch_it + 1) % len(batch_names)
    assert len(batch_images) == len(batch_labels) != 0
    return batch_images, minibatchsize, batch_it, batch_labels, new_iteration, \
        classes_true, batch_imgnames

def createMinibatch(batch_data, minibatch_it, minibatchsize):
    minibatch_data = []
    for it in range(minibatch_it * minibatchsize, min((minibatch_it + 1) * \
            minibatchsize,len(batch_data))):
        minibatch_data.append(batch_data[it])
    return np.array(minibatch_data)

def train(params):
    print("training starts now!")
    params.traintype = int(params.traintype)
    params.capacity = int(params.capacity)
    params.load_model = bool(params.load_model)
    # check if given paths really exist
    if params.phase == "training":
        assert os.path.exists(params.train_batch_dir), \
            "Train batch directory doesn't exists!"
        assert os.path.exists(params.val_batch_dir), \
            "Validation batch directory doesn't exists!"
        # load the names of the training & validation batchfiles
        train_batch_names = os.listdir(params.train_batch_dir)
        assert train_batch_names != [], \
            "Training batch directory is empty!"
        val_batch_names = os.listdir(params.val_batch_dir)
        assert val_batch_names != [], \
            "Validation batch directory is empty!"
        batch0 = np.load(params.train_batch_dir + '/' + train_batch_names[0])
    elif params.phase == "test":
        assert os.path.exists(params.test_batch_dir), \
            "Test batch directory doesn't exists!"
        test_batch_names = os.listdir(params.val_batch_dir)
        assert test_batch_names != [], \
            "Validation batch directory is empty!"
        batch0 = np.load(params.test_batch_dir + '/' + test_batch_names[0])

    #
    assert os.path.exists(params.model_dir) or not params.load_model, \
        "Model directory doesn't exists!"
    assert os.path.exists(params.vocab_path) or not params.vocab_path, \
        "Vocabulary file doesn't exists!"

    # reads the vocabulary from file
    vocabulary = open(params.vocab_path).readlines()
    vocabulary[-1] = vocabulary[-1] + '\n'
    vocabulary = [a[:-1] for a in vocabulary]
    vocabulary.append('END')

    # extract some needed paramters from the first training batch    
    num_classes = batch0['num_classes']
    max_num_tokens = len(batch0['labels'][0])

    if params.load_model:
        model = models.ModelFactory.load(params.model_dir, params.traintype, max_num_tokens)
        model.capacity = params.capacity
    else:
        # create the model directory
        if not os.path.exists(params.model_dir):
            os.makedirs(params.model_dir)
        params.model_dir = params.model_dir + '/model-' + \
            str(datetime.datetime.now())
        if not os.path.exists(params.model_dir):
            os.makedirs(params.model_dir)
        #code.interact(local=dict(globals(), **locals()))
        tf.reset_default_graph()
        model = Model(num_classes, max_num_tokens, params.model_dir, vocabulary, capacity= \
            params.capacity, train_mode=params.traintype)
    # ensures capability of loading models
    print('model stored at:')
    print(model.model_dir)
    #model = models.ModelFactory.load(model.model_dir, model.train_mode, max_num_tokens)
    # creates the first validation batch
    if params.phase == 'test':
        print('Start testing!!!')
        #
        results = model.model_dir + '/results.txt'
        fout = open(results, 'w')
        test_batch_it = 0
        test_batch_images, test_minibatchsize, test_batch_it, \
            test_batch_labels, new_test_iteration, test_batch_classes_true, \
            test_batch_imgnames = loadBatch(params.test_batch_dir, test_batch_names, \
            test_batch_it, model, model.capacity)
        num_samples = 0
        num_correct = 0
        while not new_test_iteration: # randomise!!!
            print(test_batch_it)
            test_batch_predictions = []
            for test_minibatch_it in range(test_batch_images.shape[0] \
                    / test_minibatchsize):
                # create the minibatches
                test_minibatch_images = createMinibatch(test_batch_images, \
                    test_minibatch_it, test_minibatchsize)
                test_minibatch_labels = createMinibatch(test_batch_labels, \
                    test_minibatch_it, test_minibatchsize)
                test_minibatch_imgnames = createMinibatch(test_batch_imgnames, \
                    test_minibatch_it, test_minibatchsize)
                #
                #code.interact(local=dict(globals(), **locals()))
                test_minibatch_predictions = model.predict(test_minibatch_images, \
                    test_minibatch_labels)
                for batch in range(test_minibatch_labels.shape[0]):
                    num_samples = num_samples + 1
                    line = test_minibatch_imgnames[batch] + '\n'
                    label_pred = ''
                    label_gold = ''
                    all_were_correct = 1
                    for token in range(test_minibatch_labels.shape[1]):
                        label_pred = label_pred + model.vocabulary[ \
                            test_minibatch_predictions[batch][token]] + ' '
                        #code.interact(local=dict(globals(), **locals()))
                        label_gold = label_gold + model.vocabulary[ \
                            int(test_minibatch_labels[batch][token])] + ' '
                        if test_minibatch_predictions[batch][token] != \
                                test_minibatch_labels[batch][token]:
                            all_were_correct = 0
                    num_correct = num_correct + all_were_correct
                    # code.interact(local=dict(globals(), **locals()))
                    line = line + label_gold[:-1] + '\n' + label_pred[:-1] + '\n' + '-1' + \
                        '\n' + '-1' + '\n\n'
                    print(line)
                    print(str(num_correct) + ' / ' + str(num_samples))
                    fout.write(line)
            test_batch_images, test_minibatchsize, test_batch_it, \
                test_batch_labels, new_test_iteration, test_batch_classes_true, \
                test_batch_imgnames = loadBatch(params.test_batch_dir, test_batch_names, \
                test_batch_it, model, model.capacity)
        fout.close()
    elif params.phase == 'training':
        val_batch_it = 0
        #code.interact(local=dict(globals(), **locals()))
        # 
        val_batch_images, val_minibatchsize, val_batch_it, val_batch_labels,_, \
            val_classes_true, val_batch_imgnames = loadBatch(params.val_batch_dir, \
            val_batch_names, val_batch_it, model, model.capacity)
        # validation minibatch iterator needs to be created
        val_minibatch_it = 0
        val_stats = np.zeros(model.num_classes + 1)
        # the training loop
        for epoch in range(model.nr_epochs):
            train_batch_it = 0
            # 
            train_batch_images, train_minibatchsize, train_batch_it, \
                train_batch_labels, new_train_iteration, train_batch_classes_true, \
                train_batch_imgnames = loadBatch(params.train_batch_dir, train_batch_names, \
                train_batch_it, model, model.capacity)
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

                    train_minibatch_loss_value, train_minibatch_accuracy = \
                        model.trainStep(train_minibatch_images, train_minibatch_labels)

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
                #
                #code.interact(local=dict(globals(), **locals()))
                val_minibatch_loss_value, val_minibatch_accuracy \
                    = model.valStep(val_minibatch_images, val_minibatch_labels)

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
                    train_batch_it)
                val_minibatch_it = val_minibatch_it + 1
                # 
                if (val_minibatch_it + 1) * val_minibatchsize > len(val_batch_images):
                    val_batch_images, val_minibatchsize, val_batch_it, \
                        val_batch_labels,_,val_classes_true, val_batch_imgnames = \
                        loadBatch(params.val_batch_dir, val_batch_names, val_batch_it, model,\
                        model.capacity)
                    val_minibatch_it = 0                
                train_batch_images, train_minibatchsize, train_batch_it, \
                    train_batch_labels, new_train_iteration, train_batch_classes_true, \
                    train_batch_imgnames = loadBatch(params.train_batch_dir, \
                    train_batch_names, train_batch_it, model, model.capacity)
                val_minibatch_loss_old = val_minibatch_loss_value
    else:
        print(params.phase + " is no valid phase!")
    model.session.close()

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
                        type=str, default=os.environ['TBD'],
                        help=('Path where the training \
                            batches are stored'))
    parser.add_argument('--val-batch-dir', dest='val_batch_dir',
                        type=str, default=os.environ['VBD'],
                        help=('Path where the validation \
                            batches are stored'))
    parser.add_argument('--test-batch-dir', dest='test_batch_dir',
                        type=str, default=os.environ['TESTBD'],
                        help=('Path where the test \
                            batches are stored'))
    parser.add_argument('--model-dir', dest='model_dir',
                        type=str, default=os.environ['EXP_MODEL_DIR'],
                        help=('Path where the model is stored \
                            if it already exists or \
                            should be stored if not.'))
    parser.add_argument('--capacity', dest='capacity',
                        type=str, default=100000,
                        help=('The total amount of floats \
                            that can be propagated through \
                            the network at a time. \
                            Influences the batch size!'))
    parser.add_argument('--vocab-path', dest='vocab_path',
                        type=str, default=os.environ['VOCABS'],
                        help=('The total amount of floats \
                            that can be propagated through \
                            the network at a time. \
                            Influences the batch size!'))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    params = process_args(args)
    train(params)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')