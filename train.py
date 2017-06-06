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
    network = tflearn.layers.conv.conv_2d(network, 64, 3, activation='relu') # padding???
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
    batchsize = tf.shape(network)[0]
    network = tf.transpose(network, [1,0,2,3])    
    rnncell_fw = tf.contrib.rnn.LSTMCell(128) # TODO choose parameter
    # code.interact(local=locals())
    fw_state = rnncell_fw.zero_state(batch_size=batchsize, dtype=tf.float32)
    rnncell_bw = tf.contrib.rnn.LSTMCell(128) # TODO choose parameter
    bw_state = rnncell_bw.zero_state(batch_size=batchsize, dtype=tf.float32)
    l = tf.TensorArray(dtype=tf.float32, size=tf.shape(network)[0])
    params = [tf.constant(0), network, l, fw_state, bw_state]
    while_condition = lambda i, network, l, fw_state, bw_state: tf.less(i, tf.shape(network)[0])
    def body(i, network, l, fw_state, bw_state):
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(rnncell_fw, rnncell_bw, network[i], initial_state_fw=fw_state,initial_state_bw=bw_state) #,dtype=tf.float32)
        fw_state, bw_state = output_states
        code.interact(local=locals())
        l = l.write(i, outputs)        
        return [tf.add(i, 1), network, l, fw_state, bw_state]
    i, network, l, fw_state, bw_state = tf.while_loop(while_condition, body, params)
    return l.stack(), i #

class Model:
    def __init__(self, batchsize):
        self.nr_epochs = 1
        self.batchsize = batchsize
        # H,W,batchsize, was bedeutet input_data???
        network = tflearn.layers.core.input_data(shape=[None, None, None, 1])
        self.input_var = network
        # How do I test parts of the network??
        network = createCNNModel(network)
        self.convshape = network
        network, aaa = createEncoderLSTM(network)
        self.encshape = tf.shape(network)
        self.aaa = aaa
        self.aaaNetwork = network

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
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    if phase == "training": # epochs!!!
        for i in range(model.nr_epochs):
            for batchfile in batchfiles: # randomise!!!
                print('load ' + batchfile + '!')
                batch = np.load(batch_dir + '/' + batchfile)
                images = batch['images']
                labels = batch['labels']
                if len(images) != 0:
                    # code.interact(local=locals())
                    # sess.run(model.aaaNetwork, feed_dict={model.input_var:images})
                    print(images.shape)
                    pred = np.array(model.model.predict(images))
                    print(model.convshape.shape)
                    print(pred.shape)
                    # print(model.aaa)
                    #sess.run(model.aaa, feed_dict={model.input_var:images})
                # model.fit(batch) 
    sess.close()           

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')