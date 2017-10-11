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

def loadBatch(batch_dir, batch_names, batch_it, model):
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
            model.capacity)
        if minibatchsize != 0:
            break
        else:
            print('Pictures to big to process!')
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

def iterateOneEpoch(phase, batch_dir, batch_names, model):
    if model.used_loss == 'label':
        results = model.model_dir + '/' + phase + str(model.current_epoch) + '_results.txt'
        shutil.rmtree(results, ignore_errors=True)
        fout = open(results, 'w')
    else:
        false2false = np.zeros(model.num_classes)
        false2true = np.zeros(model.num_classes)
        true2true = np.zeros(model.num_classes)
        true2false = np.zeros(model.num_classes)
    batch_it = 0
    batch_images, minibatchsize, batch_it, \
        batch_labels, new_iteration, batch_classes_true, \
        batch_imgnames = loadBatch(batch_dir, batch_names, \
        batch_it, model)
    num_samples = 0
    samples_correct = 0
    tokens_correct = 0
    num_tokens = 0
    batch_losses = []
    batch_accuracies = []
    while not new_iteration: # randomise!!!
        print(batch_it)
        batch_predictions = []
        for minibatch_it in range(batch_images.shape[0] \
                / minibatchsize):
            # create the minibatches
            minibatch_images = createMinibatch(batch_images, \
                minibatch_it, minibatchsize)
            minibatch_labels = createMinibatch(batch_labels, \
                minibatch_it, minibatchsize)
            minibatch_imgnames = createMinibatch(batch_imgnames, \
                minibatch_it, minibatchsize)
            #
            if phase == 'train':
                minibatch_loss_value, minibatch_accuracy,minibatch_predictions = \
                    model.trainStep(minibatch_images, minibatch_labels)
            else:                
                minibatch_loss_value, minibatch_accuracy,minibatch_predictions = \
                    model.valStep(minibatch_images, minibatch_labels)

            print(phase + 'minibatch(' + str(model.current_epoch) + ',' + str(batch_it) \
                + ',' + str(minibatch_it*minibatchsize)+') : '+\
                str(minibatch_images.shape) + ' : ' + \
                str(minibatch_labels.shape) + ' : ' + \
                str(minibatch_loss_value) + ' : ' + \
                str(model.current_train_accuracy) + ' : ' + \
                str(model.current_infer_accuracy) + ' : ' + \
                str(np.mean(batch_accuracies)) + ' : ' + \
                model.train_mode)
            batch_losses.append(minibatch_loss_value)
            batch_accuracies.append(model.current_infer_accuracy)
            if phase != 'train':
                label_pred = ''
                label_gold = ''
                for batch in range(minibatch_labels.shape[0]):
                    num_samples = num_samples + 1
                    all_were_correct = 1
                    for token in range(minibatch_labels.shape[1]):
                        num_tokens = num_tokens + 1
                        if minibatch_predictions[batch][token] != \
                                minibatch_labels[batch][token]:
                            all_were_correct = 0
                        else:
                            tokens_correct = tokens_correct + 1
                    samples_correct = samples_correct + all_were_correct
                    if model.used_loss == 'label':
                        line = minibatch_imgnames[batch] + '\t'
                        label_pred = ''
                        label_gold = ''
                        for token in range(minibatch_labels.shape[1]):
                            if len(model.vocabulary) == \
                                    minibatch_predictions[batch][token]+1:
                                break
                            label_pred = label_pred + model.vocabulary[ \
                                minibatch_predictions[batch][token]] + ' '
                            #code.interact(local=dict(globals(), **locals()))
                        for token in range(minibatch_labels.shape[1]):
                            if len(model.vocabulary)== \
                                int(minibatch_labels[batch][token])+1:
                                break
                            label_gold = label_gold + model.vocabulary[ \
                                int(minibatch_labels[batch][token])] + ' '                        
                        line = line + label_gold[:-1] + '\t' + label_pred[:-1] + '\t' + \
                            '-1' + '\t' + '-1' + '\n'
                        
                        fout.write(line)
                    else:
                        for token in range(minibatch_labels.shape[1]):
                            if minibatch_labels[batch][token] == 0 and \
                                    minibatch_predictions[batch][token] == 0:
                                false2false[token] = false2false[token] + 1
                            elif minibatch_labels[batch][token] == 0 and \
                                    minibatch_predictions[batch][token] == 1:
                                false2true[token] = false2true[token] + 1
                            elif minibatch_labels[batch][token] == 1 and \
                                    minibatch_predictions[batch][token] == 1:
                                true2true[token] = true2true[token] + 1
                            elif minibatch_labels[batch][token] == 1 and \
                                    minibatch_predictions[batch][token] == 0:
                                true2false[token] = true2false[token] + 1
                            else:
                                print('error!!!')
                                code.interact(local=dict(globals(), **locals()))
                    '''if batch % 10 == 0:
                        absolute_acc = float(samples_correct) / float(num_samples)
                        token_acc = float(tokens_correct) / float(num_tokens)
                        print('abs: ' + str(absolute_acc) + ', tok: ' + str(token_acc))'''
                print(str(samples_correct) + ' / ' + str(num_samples))
                print(label_gold[:-1])
                print('')
                print(label_pred[:-1])
                print('')
            #elif minibatch_accuracy > 0.5:
            #    return np.mean(batch_losses), np.mean(batch_accuracies)
        batch_images, minibatchsize, batch_it, \
            batch_labels, new_iteration, batch_classes_true, \
            batch_imgnames = loadBatch(batch_dir, batch_names, \
            batch_it, model)
    if phase != 'train':
        absolute_acc = float(samples_correct) / float(num_samples)
        absolute_acc_path = model.model_dir + '/' + phase + str(model.current_epoch) + \
            '_absolute_accuracy.txt'
        shutil.rmtree(absolute_acc_path, ignore_errors=True)
        absolute_acc_writer = open(absolute_acc_path, 'w')
        absolute_acc_writer.write(str(absolute_acc))
        absolute_acc_writer.close()
        token_acc = float(tokens_correct) / float(num_tokens)
        token_acc_path = model.model_dir + '/' + phase + str(model.current_epoch) + \
            '_token_accuracy.txt'
        shutil.rmtree(token_acc_path, ignore_errors=True)
        token_acc_writer = open(token_acc_path, 'w')
        token_acc_writer.write(str(token_acc))
        token_acc_writer.close()
        print('abs: ' + str(absolute_acc) + ', tok: ' + str(token_acc))
        if model.used_loss == 'label':
            fout.close()
        else:
            with open(model.model_dir + '/confusion.npz', 'w') as fout:
                np.savez(fout,false2false=false2false,false2true=false2true, \
                    true2true=true2true,true2false=true2false)
            stats = {}
            for c in range(model.num_classes):
                num_right = float(false2false[c] + true2true[c])
                num = float(false2false[c] + true2true[c] + false2true[c] + true2false[c])
                n = true2true[c] + true2false[c]
                class_accuracy = num_right / num
                f2f = false2false[c] / num
                t2t = true2true[c] / num
                f2t = false2true[c] / num
                t2f = true2false[c] / num
                stats.update({(class_accuracy,c):(model.vocabulary[c], str(f2f), \
                    str(t2t), str(f2t), str(t2f), str(num))})
            keys = sorted(stats.keys())
            s = ''
            for key in keys:
                c, f2f, t2t, f2t, t2f, n = stats[key]
                s = s +'\"'+c+'\" : '+str(key[0])+' : ('+f2f+','+t2t+','+f2t+','+ \
                    t2f+') : ' + n + '\n'
            stats_path = model.model_dir + '/stats.txt'
            shutil.rmtree(stats_path, ignore_errors=True)
            stats_writer = open(stats_path, 'w')
            stats_writer.write(s)
            stats_writer.close()
    return np.mean(batch_losses), np.mean(batch_accuracies)


def train(params):
    try:
        print("training starts now!")
        #code.interact(local=dict(globals(), **locals()))
        #params.trainmode = int(params.trainmode)
        if params.capacity != None:
            params.capacity = int(params.capacity)
        params.load_model = bool(params.load_model)
        if params.load_model and not params.combine_models:
            if params.model_dir == None:
                #code.interact(local=dict(globals(), **locals()))
                read_path = '/home/sbender/tmp/' + params.trainmode + '.tmp'
                if not os.path.exists(read_path):
                    print(read_path + ' does not exist!')
                    quit()
                reader = open(read_path, 'r')
                params.model_dir = reader.read()
        else:
            params.model_dir = os.environ['EXP_MODEL_DIR']
        if params.combine_models and params.fe_dir == None:
            tm = params.trainmode.split('_')
            tm = tm[0] + '___' + tm[3]
            read_path = '/home/sbender/tmp/' + tm + '.tmp'
            if not os.path.exists(read_path):
                print(read_path + ' does not exist!')
                quit()
            reader = open(read_path, 'r')
            params.fe_dir = reader.read()
        if params.combine_models and params.enc_dec_dir == None:
            tm = params.trainmode.split('_')
            tm = tm[0] + '_' + tm[1] + '_' + tm[2] + '_'
            read_path = '/home/sbender/tmp/' + tm + '.tmp'
            if not os.path.exists(read_path):
                print(read_path + ' does not exist!')
                quit()
            reader = open(read_path, 'r')
            params.enc_dec_dir = reader.read()
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
            test_batch_names = os.listdir(params.test_batch_dir)
            assert test_batch_names != [], \
                "Validation batch directory is empty!"
            batch0 = np.load(params.test_batch_dir + '/' + test_batch_names[0])

        #
        assert os.path.exists(params.model_dir) or not params.load_model, \
            "Model directory doesn't exists!"
        assert os.path.exists(params.vocab_path) or not params.vocab_path, \
            "Vocabulary file doesn't exists!"

        # reads the vocabulary from file
        vocabulary = open(params.vocab_path).read() + '\n' + 'END'

        # extract some needed paramters from the first training batch    
        num_classes = batch0['num_classes']
        max_num_tokens = len(batch0['labels'][0])
        num_features = batch0['images'].shape[3]

        #code.interact(local=dict(globals(), **locals()))
        if params.load_model:
            model = models.ModelFactory.load(params.model_dir,params.trainmode,max_num_tokens,\
                params.learning_rate, capacity=params.capacity, \
                combine_models=params.combine_models, \
                fe_dir=params.fe_dir, enc_dec_dir=params.enc_dec_dir)
        else:
            tf.reset_default_graph()
            model = Model(params.model_dir, num_classes, max_num_tokens, vocabulary, capacity= \
                params.capacity, train_mode=params.trainmode, learning_rate=params.learning_rate, \
                num_features=num_features)
        # code.interact(local=dict(globals(), **locals()))
        # ensures capability of loading models
        print('model stored at:')
        print(model.model_dir)
        #code.interact(local=dict(globals(), **locals()))
        #model = models.ModelFactory.load(model.model_dir, model.train_mode, max_num_tokens)
        # creates the first validation batch
        if params.phase == 'test':
            test_loss, test_accuracy = iterateOneEpoch('test', params.test_batch_dir, \
                test_batch_names, model, 0)
        elif params.phase == 'training':
            # the training loop
            train_losses = [float('inf')]
            val_losses = [float('inf')]
            while True:
                train_loss, train_accuracy = iterateOneEpoch('train', params.train_batch_dir,\
                    train_batch_names, model)
                val_loss, val_accuracy = iterateOneEpoch('val', params.val_batch_dir, \
                    val_batch_names, model)
                model.current_epoch = model.current_epoch + 1
                #print('validation phase')
                #code.interact(local=dict(globals(), **locals()))
                if train_losses[-1] < train_loss:
                    model.decayLearningRate(0.5)
                elif val_losses[-1] < val_loss and model.current_epoch > 5:
                    print('starts to overfit!')
                    write_path = '/home/sbender/tmp/' + model.train_mode + '.tmp'
                    #shutil.rmtree(write_path)
                    writer = open(write_path, 'w')
                    writer.write(model.model_dir)
                    print('before termination')
                    code.interact(local=dict(globals(), **locals()))
                    break
                train_losses.append(train_loss)
                val_losses.append(val_loss)

        else:
            print(params.phase + " is no valid phase!")
        model.session.close()
    except:
        print('something went wrong!')
        print(sys.exc_info())
        code.interact(local=dict(globals(), **locals()))

def process_args(args):
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--phase', dest='phase',
                        type=str, default='training',
                        help=('The phase (training or prediction).'))
    parser.add_argument('--trainmode', dest='trainmode',
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
                        type=str, default=None,
                        help=('Path where the model is stored \
                            if it already exists or \
                            should be stored if not.'))
    parser.add_argument('--capacity', dest='capacity',
                        type=str, default=None,
                        help=('The total amount of floats \
                            that can be propagated through \
                            the network at a time. \
                            Influences the batch size!'))
    parser.add_argument('--learning_rate', dest='learning_rate',
                        type=str, default=None,
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
    parser.add_argument('--combine_models', dest='combine_models',
                        type=bool, default=False,
                        help=('The total amount of floats \
                            that can be propagated through \
                            the network at a time. \
                            Influences the batch size!'))
    parser.add_argument('--fe_dir', dest='fe_dir',
                        type=str, default=None,
                        help=('The total amount of floats \
                            that can be propagated through \
                            the network at a time. \
                            Influences the batch size!'))
    parser.add_argument('--enc_dec_dir', dest='enc_dec_dir',
                        type=str, default=None,
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

'''if params.phase == 'test':
        print('Start testing!!!')
        #
        if model.used_loss == 'label':
            results = model.model_dir + '/results.txt'
            shutil.rmtree(results, ignore_errors=True)
            fout = open(results, 'w')
        else:
            false2false = np.zeros(model.num_classes)
            false2true = np.zeros(model.num_classes)
            true2true = np.zeros(model.num_classes)
            true2false = np.zeros(model.num_classes)
        test_batch_it = 0
        test_batch_images, test_minibatchsize, test_batch_it, \
            test_batch_labels, new_test_iteration, test_batch_classes_true, \
            test_batch_imgnames = loadBatch(params.test_batch_dir, test_batch_names, \
            test_batch_it, model, model.capacity)
        num_samples = 0
        samples_correct = 0
        tokens_correct = 0
        num_tokens = 0
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
                    all_were_correct = 1
                    for token in range(test_minibatch_labels.shape[1]):
                        num_tokens = num_tokens + 1
                        if test_minibatch_predictions[batch][token] != \
                                test_minibatch_labels[batch][token]:
                            all_were_correct = 0
                        else:
                            tokens_correct = tokens_correct + 1
                    samples_correct = samples_correct + all_were_correct
                    if model.used_loss == 'label':
                        line = test_minibatch_imgnames[batch] + '\t'
                        label_pred = ''
                        label_gold = ''
                        for token in range(test_minibatch_labels.shape[1]):
                            if len(model.vocabulary) == \
                                    test_minibatch_predictions[batch][token]+1:
                                break
                            label_pred = label_pred + model.vocabulary[ \
                                test_minibatch_predictions[batch][token]] + ' '
                            #code.interact(local=dict(globals(), **locals()))
                        for token in range(test_minibatch_labels.shape[1]):
                            if len(model.vocabulary)== \
                                int(test_minibatch_labels[batch][token])+1:
                                break
                            label_gold = label_gold + model.vocabulary[ \
                                int(test_minibatch_labels[batch][token])] + ' '                        
                        line = line + label_gold[:-1] + '\t' + label_pred[:-1] + '\t' + \
                            '-1' + '\t' + '-1' + '\n'
                        print(model.train_mode)
                        print(label_gold[:-1])
                        print('')
                        print(label_pred[:-1])
                        print('')
                        print(str(samples_correct) + ' / ' + str(num_samples))
                        fout.write(line)
                    else:
                        for token in range(test_minibatch_labels.shape[1]):
                            if test_minibatch_labels[batch][token] == 0 and \
                                    test_minibatch_predictions[batch][token] == 0:
                                false2false[token] = false2false[token] + 1
                            elif test_minibatch_labels[batch][token] == 0 and \
                                    test_minibatch_predictions[batch][token] == 1:
                                false2true[token] = false2true[token] + 1
                            elif test_minibatch_labels[batch][token] == 1 and \
                                    test_minibatch_predictions[batch][token] == 1:
                                true2true[token] = true2true[token] + 1
                            elif test_minibatch_labels[batch][token] == 1 and \
                                    test_minibatch_predictions[batch][token] == 0:
                                true2false[token] = true2false[token] + 1
                            else:
                                print('error!!!')
                                code.interact(local=dict(globals(), **locals()))
                    if batch % 10 == 0:
                        absolute_acc = float(samples_correct) / float(num_samples)
                        token_acc = float(tokens_correct) / float(num_tokens)
                        print('abs: ' + str(absolute_acc) + ', tok: ' + str(token_acc))
            test_batch_images, test_minibatchsize, test_batch_it, \
                test_batch_labels, new_test_iteration, test_batch_classes_true, \
                test_batch_imgnames = loadBatch(params.test_batch_dir, test_batch_names, \
                test_batch_it, model, model.capacity)
        absolute_acc = float(samples_correct) / float(num_samples)
        absolute_acc_path = model.model_dir + '/absolute_accuracy.txt'
        shutil.rmtree(absolute_acc_path, ignore_errors=True)
        absolute_acc_writer = open(absolute_acc_path, 'w')
        absolute_acc_writer.write(str(absolute_acc))
        absolute_acc_writer.close()
        token_acc = float(tokens_correct) / float(num_tokens)
        token_acc_path = model.model_dir + '/token_accuracy.txt'
        shutil.rmtree(token_acc_path, ignore_errors=True)
        token_acc_writer = open(token_acc_path, 'w')
        token_acc_writer.write(str(token_acc))
        token_acc_writer.close()
        print('abs: ' + str(absolute_acc) + ', tok: ' + str(token_acc))
        if model.used_loss == 'label':
            fout.close()
        else:
            with open(model.model_dir + '/confusion.npz', 'w') as fout:
                np.savez(fout,false2false=false2false,false2true=false2true, \
                    true2true=true2true,true2false=true2false)
            stats = {}
            for c in range(model.num_classes):
                num_right = float(false2false[c] + true2true[c])
                num = float(false2false[c] + true2true[c] + false2true[c] + true2false[c])
                n = true2true[c] + true2false[c]
                class_accuracy = num_right / num
                f2f = false2false[c] / num
                t2t = true2true[c] / num
                f2t = false2true[c] / num
                t2f = true2false[c] / num
                stats.update({(class_accuracy,c):(model.vocabulary[c], str(f2f), \
                    str(t2t), str(f2t), str(t2f), str(num))})
            keys = sorted(stats.keys())
            s = ''
            for key in keys:
                c, f2f, t2t, f2t, t2f, n = stats[key]
                s = s +'\"'+c+'\" : '+str(key[0])+' : ('+f2f+','+t2t+','+f2t+','+ \
                    t2f+') : ' + n + '\n'
            stats_path = model.model_dir + '/stats.txt'
            shutil.rmtree(stats_path, ignore_errors=True)
            stats_writer = open(stats_path, 'w')
            stats_writer.write(s)
            stats_writer.close()
    elif params.phase == 'training':
        # the training loop
        last_loss = float("inf")
        while current_loss < last_loss:
            train_batch_it = 0
            # 
            train_batch_images, train_minibatchsize, train_batch_it, \
                train_batch_labels, new_train_iteration, train_batch_classes_true, \
                train_batch_imgnames = loadBatch(params.train_batch_dir, train_batch_names, \
                train_batch_it, model, model.capacity)
            val_minibatch_loss_old = float("inf")
            train_stats = np.zeros(model.num_classes + 1)
            train_batch_accuracies = []
            train_batch_losses = []
            while not new_train_iteration: # randomise!!!
                early = False
                print(train_batch_it)
                for train_minibatch_it in range(train_batch_images.shape[0] \
                        / train_minibatchsize):
                    # create the minibatches
                    train_minibatch_images = createMinibatch(train_batch_images, \
                        train_minibatch_it, train_minibatchsize)
                    train_minibatch_labels = createMinibatch(train_batch_labels, \
                        train_minibatch_it, train_minibatchsize)

                    train_minibatch_loss_value, train_minibatch_accuracy = \
                        model.trainStep(train_minibatch_images, train_minibatch_labels)

                    print('trainminibatch(' + str(epoch) + ',' + str(train_batch_it) \
                        + ',' + str(train_minibatch_it*train_minibatchsize)+') : '+\
                        str(train_minibatch_images.shape) + ' : ' + \
                        str(train_minibatch_labels.shape) + ' : ' + \
                        str(train_minibatch_loss_value) + ' : ' + \
                        str(train_minibatch_accuracy) + ' : ' + \
                        str(np.mean(train_batch_accuracies)) + ' : ' + \
                        model.train_mode)
                    train_batch_losses.append(train_minibatch_loss_value)
                    train_batch_accuracies.append(train_minibatch_accuracy)
                    if train_minibatch_accuracy > 0.7:
                        print('early stop!')
                        early = True
                        #code.interact(local=dict(globals(), **locals()))
                        #write_path = '/home/sbender/tmp/' + model.train_mode + '.tmp'
                        #writer = open(write_path, 'w')
                        #writer.write(model.model_dir)
                        #model.session.close()
                        break
                train_batch_images, train_minibatchsize, train_batch_it, \
                    train_batch_labels, new_train_iteration, train_batch_classes_true, \
                    train_batch_imgnames = loadBatch(params.train_batch_dir, \
                    train_batch_names, train_batch_it, model, model.capacity)
                if early:
                    break
                #print('in train iteration')
                #code.interact(local=dict(globals(), **locals()))
            val_batch_it = 0
            # 
            val_batch_images, val_minibatchsize, val_batch_it, \
                val_batch_labels, new_val_iteration, val_batch_classes_true, \
                val_batch_imgnames = loadBatch(params.val_batch_dir, val_batch_names, \
                val_batch_it, model, model.capacity)
            val_minibatch_loss_old = float("inf")
            val_stats = np.zeros(model.num_classes + 1)
            val_batch_losses = []
            val_batch_accuracies = []
            while not new_val_iteration: # randomise!!!
                print(val_batch_it)
                for val_minibatch_it in range(val_batch_images.shape[0] \
                        / val_minibatchsize):
                    # create the minibatches
                    val_minibatch_images = createMinibatch(val_batch_images, \
                        val_minibatch_it, val_minibatchsize)
                    val_minibatch_labels = createMinibatch(val_batch_labels, \
                        val_minibatch_it, val_minibatchsize)

                    #print('in val iteration')
                    #code.interact(local=dict(globals(), **locals()))
                    val_minibatch_loss_value, val_minibatch_accuracy = \
                        model.valStep(val_minibatch_images, val_minibatch_labels)

                    print('valminibatch(' + str(epoch) + ',' + str(val_batch_it) \
                        + ',' + str(val_minibatch_it*val_minibatchsize)+') : '+\
                        str(val_minibatch_images.shape) + ' : ' + \
                        str(val_minibatch_labels.shape) + ' : ' + \
                        str(val_minibatch_loss_value) + ' : ' + \
                        str(val_minibatch_accuracy) + ' : ' + \
                        model.train_mode)
                    val_batch_losses.append(val_minibatch_loss_value)
                    val_batch_accuracies.append(val_minibatch_accuracy)
                val_batch_images, val_minibatchsize, val_batch_it, \
                    val_batch_labels, new_val_iteration, val_batch_classes_true, \
                    val_batch_imgnames = loadBatch(params.val_batch_dir, val_batch_names, \
                    val_batch_it, model, model.capacity)
            # adds a log
            model.addLog(np.mean(train_batch_losses), np.mean(val_batch_losses), \
                np.mean(train_batch_accuracies), np.mean(val_batch_accuracies), epoch)
            model.current_epoch = model.current_epoch + 1
            #print('validation phase')
            #code.interact(local=dict(globals(), **locals()))
            if last_loss > np.mean(val_batch_losses):
                last_loss = np.mean(val_batch_losses)
            else:
                print('early stop!')
                write_path = '/home/sbender/tmp/' + model.train_mode + '.tmp'
                #shutil.rmtree(write_path)
                writer = open(write_path, 'w')
                writer.write(model.model_dir)
                break'''