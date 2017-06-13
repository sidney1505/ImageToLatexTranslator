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
                model.num_classes = batch['num_classes'] # need to be known earlier!!!
                if len(images) != 0:
                    # code.interact(local=locals())
                    # sess.run(model.aaaNetwork, feed_dict={model.input_var:images})
                    for j in range(len(images)):
                        print(images.shape)
                        # code.interact(local=locals())
                        pred = np.array(model.model.predict(np.take(images,range(i,min(len(images)-1,i+model.batchsize)),axis=0)))
                        print(pred.shape)
                    # print(model.aaa)
                    #sess.run(model.aaa, feed_dict={model.input_var:images})
                # model.fit(batch) 
    sess.close()           

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')