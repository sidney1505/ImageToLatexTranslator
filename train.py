import sys, os, argparse, logging
import numpy as np
import code
import PIL
from PIL import Image
# from model.model import Model
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

def process_args(args):
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--phase', dest='phase',
                        type=str, default='training',
                        help=('Directory containing processed images.'))
    parser.add_argument('--batch-dir', dest='batch_dir',
                        type=str, required=True,
                        help=('path where the batches are stored'))
    parser.add_argument('--batch-size', dest='batch_size',
                        type=str, default=5,
                        help=('size of the minibatches'))
    parameters = parser.parse_args(args)
    return parameters

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def createCNNModel(network):
    network = conv_2d(network, 64, 3, activation='relu') # padding???
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = tflearn.layers.normalization.batch_normalization(network) #same as torch?

    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, [1,2], strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = tflearn.layers.normalization.batch_normalization(network) #same as torch?
    network = max_pool_2d(network, [2,1], strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = tflearn.layers.normalization.batch_normalization(network) #same as torch?

    return network

def createEncoderLSTM(network):
    network = tf.transpose(network, [1,0,2,3])


    
    rnncell_fw = tf.contrib.rnn.LSTMCell(128) # TODO choose parameter
    rnncell_bw = tf.contrib.rnn.LSTMCell(128) # TODO choose parameter
    # initial_state_bw = tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(128,tf.float32)
    # code.interact(local=locals())
    net = []
    i = tf.constant(0)
    while_condition = lambda i: tf.less(i, tf.shape(network)[0])
    def body(i):
        print('hi')
        print(i)
        print(tf.shape(network)[0])
        net = [tf.nn.bidirectional_dynamic_rnn(rnncell_fw, rnncell_bw, network[i], dtype=tf.float32)]
        # sequence_length=length(network[i]), initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw)
        print('bbb')
        return [tf.add(i, 1)]


    aaa = tf.while_loop(while_condition, body, [i])
    code.interact(local=locals())
        #code.interact(local=locals())
    #network = tf.constant(np.zeros((4,2,5)))
    #network = tf.nn.bidirectional_dynamic_rnn(rnncell_fw, rnncell_bw, network, dtype=tf.float64)
        # sequence_length=length(network), 
            # initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw
            #dtype=rnncell_fw.state_size
            
    #i = tf.constant(0)
    #while_condition = lambda i: tf.less(i, tf.shape(network)[0])
    #rnncell_fw = tflearn.layers.recurrent.BasicLSTMCell(128)
    #rnncell_bw = tflearn.layers.recurrent.BasicLSTMCell(128)
    #code.interact(local=locals())
    #def body(i):
    #    network[i] = tflearn.layers.recurrent.bidirectional_rnn(network[i], rnncell_fw, rnncell_bw, return_seq=True, dynamic=True)
    #    return [tf.add(i, 1)]

    #tf.while_loop(while_condition, body, [i])

    #i = 0
    #MAX_HEIGHT = 1000
    #real_height = 50

    #axes = [1, 0] + list(range(2, 3)) # 3 is number of axis of input
    #inference = tf.transpose(network[i], (axes))
    #inference = tf.unstack(inference, num=MAX_HEIGHT)
    #output = inference[:real_height] # output for each i

    # return network
    return network, aaa

class Model:
    def __init__(self, batchsize):
        self.nr_epochs = 1
        self.batchsize = batchsize
        # H,W,batchsize, was bedeutet input_data???
        network = tflearn.layers.core.input_data(shape=[None, None, None, 1])
        self.input_var = network
        # How do I test parts of the network??
        network = createCNNModel(network)
        network, aaa = createEncoderLSTM(network)
        self.aaa = aaa

        self.model = tflearn.DNN(network)

    def fit(self, batch):
        self.model.fit(batch['images'], batch['labels'], self.nr_epochs, self.batchsize)



def main(args):
    parameters = process_args(args)
    print("main")
    phase = parameters.phase
    batch_dir = parameters.batch_dir
    assert os.path.exists(batch_dir), batch_dir
    batchfiles = os.listdir(batch_dir)
    batchsize = parameters.batch_size
    model = Model(batchsize)
    if phase == "training": # epochs!!!
        for i in range(model.nr_epochs):
            for batchfile in batchfiles: # randomise!!!
                batch = np.load(batch_dir + '/' + batchfile)
                images = batch['images']
                labels = batch['labels']
                if len(images) != 0:
                    code.interact(local=locals())
                    model.aaa
                    pred = np.array(model.model.predict(images))
                    print(pred.shape)
                    print(model.aaa)
                # model.fit(batch)            

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')