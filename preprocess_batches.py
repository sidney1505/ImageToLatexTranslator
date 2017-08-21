import sys, os, argparse, logging, json, shutil, code
import numpy as np
import tensorflow as tf
from PIL import Image
from models.ModelFactory import Model
import models.ModelFactory

def preprocess(params):    
    print("preprocessing starts now!")
    # the directory, where the training batches are stored
    if params.traintype != None:
        read_path = '/home/sbender/tmp/' + params.traintype + '.tmp'
        if not os.path.exists(read_path):
            print(read_path + ' does not exist!')
            quit()
        reader = open(read_path, 'r')
        model_dir = reader.read()
    else:
        model_dir = params.model_dir
    batch_dir = params.batch_dir
    assert os.path.exists(batch_dir), batch_dir
    batchfiles = os.listdir(batch_dir)
    #
    assert os.path.exists(model_dir), model_dir
    #
    output_dir = params.output_dir
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    #
    batch0 = np.load(batch_dir + '/' + batchfiles[0])
    max_num_tokens = len(batch0['labels'][0])
    model = models.ModelFactory.load(model_dir, max_num_tokens)
    for batchfile in batchfiles:
        batch = np.load(batch_dir + '/' + batchfile)
        preds = []
        images = batch['images']
        train_k = 20
        while images.shape[1] * images.shape[2] * train_k > 1000000:
            train_k = train_k - 1
        if train_k == 0:
            continue
        model.minibatchsize = train_k
        for j in range(len(images) / model.minibatchsize + 1):
            imgs = []
            for b in range(j*model.minibatchsize, min((j+1)*model.minibatchsize, \
                    len(images))):
                imgs.append(images[b])
            imgs = np.array(imgs)
            feed_dict={model.input: imgs}
            features = model.session.run(model.features, feed_dict=feed_dict)
            preds.append(features)
            if (j+1) * model.minibatchsize == len(images):
                break
        preds = np.concatenate(preds)
        assert preds.shape[0] == batch['labels'].shape[0]
        with open(output_dir + "/" + batchfile, 'w') as fout:
            np.savez(fout, images=preds, labels=batch['labels'], \
                num_classes=batch['num_classes'], \
                contained_classes_list=batch['contained_classes_list'],
                image_names=batch['image_names'])
        print(batchfile + " saved! " + str(preds.shape) + " : " + str(batch['labels'].shape))
    model.session.close()

def process_args(args):
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--traintype', dest='traintype',
                        type=str, default=None,
                        help=('Directory containing model.'
                        ))
    parser.add_argument('--model-path', dest='model_dir',
                        type=str, default=None,
                        help=('Directory containing model.'
                        ))
    parser.add_argument('--batch-dir', dest='batch_dir',
                        type=str, default=os.environ['TBD'],
                        help=('path where the batches are stored'))
    parser.add_argument('--output-dir', dest='output_dir',
                        type=str, default=os.environ['TBDENC'],
                        help=('path where the new batches will be stored'))
    params = parser.parse_args(args)
    return params

def main(args):
    params = process_args(args)
    #
    preprocess(params)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
