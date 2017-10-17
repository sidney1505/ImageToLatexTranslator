import sys, os, code
import Trainer
import tensorflow as tf

class GreedyHyperparamterTuner:
    def __init__(self, best_param_dir):
        self.modes=['e2e','stepbystep']
        self.feature_extractors=['alexnetFe','wysiwysgFe','vggFe','resnetFe']
        self.encoders=['monorowEnc','birowEnc','monocolEnc','bicolEnc','quadroEnc']
        self.decoders=['simpleDec','simplegruDec','bahdanauDec','luongDec']
        self.encoder_sizes=[512,1024,2048,4096]
        self.decoder_sizes=[512,1024,2048,4096]
        self.optimizers=['sgd','momentum','adam','adadelta']
        #
        self.trainer = Trainer.Trainer(os.environ['BASE_MODEL_DIR'], os.environ['DATA_DIR'],\
            os.environ['TMP_DIR'], 1000000)
        self.best_param_dir = os.environ['BEST_MODEL_DIR']
        self.current_mode = self.readParam('mode')
        self.current_feature_extractor = self.readParam('feature_extractor')
        self.current_encoder = self.readParam('encoder')
        self.current_decoder = self.readParam('decoder')
        self.current_encoder_size = int(self.readParam('encoder_size'))
        self.current_decoder_size = int(self.readParam('decoder_size'))
        self.current_optimizer = self.readParam('optimizer')
        self.current_best_accuracy = float(self.readParam('best_accuracy'))

    def readParam(self, param):
        read_path = self.model_dir + '/' + param
        if not os.path.exists(read_path):
            raise Exception(read_path + ' does not exist!')
        reader = open(read_path, 'r')
        value = reader.read()
        reader.close()
        return value

    def writeParam(self, param, value):
        param_path = self.model_dir
        if not os.path.exists(param_path):
            os.makedirs(param_path)
        write_path = params_path + '/' + param
        shutil.rmtree(write_path, ignore_errors=True)
        writer = open(write_path, 'w')
        writer.write(str(value))
        writer.close()

    def testMode(self, mode):
        trainer.setModelParameters(mode, self.current_feature_extractor, \
            self.current_encoder, self.current_decoder, self.current_encoder_size, \
            self.current_decoder_size, self.current_optimizer)
        trainer.trainModel()
        model_accuracy = trainer.testModel()
        if self.current_best_accuracy < model_accuracy:
            self.writeParam('best_accuracy', model_accuracy)
            self.writeParam('mode', mode)

    def testFeatureExtractor(self, feature_extractor):
        trainer.setModelParameters(self.current_mode, feature_extractor, \
            self.current_encoder, self.current_decoder, self.current_encoder_size, \
            self.current_decoder_size, self.current_optimizer)
        trainer.trainModel()
        model_accuracy = trainer.testModel()
        if self.current_best_accuracy < model_accuracy:
            self.writeParam('best_accuracy', model_accuracy)
            self.writeParam('feature_extractor', feature_extractor)


    def findBestConfiguration(self):
        self.current_mode=0
        self.current_feature_extractor=0
        self.current_encoder=0
        self.current_decoder=0
        self.current_encoder_size=0
        self.current_decoder_size=0
        self.current_optimizer=0
        best_accuracy=0.0
        # find the best mode
        trainer = Trainer.Trainer()
        for mode in range(len(self.modes)):
            trainer.createModel(
                self.modes[mode], 
                self.feature_extractors[self.current_feature_extractor], \
                self.encoders[self.current_encoder], \
                self.decoders[self.current_decoder], \
                self.encoder_sizes[self.current_encoder_size], \
                self.decoder_sizes[self.current_decoder_size], \
                self.optimizers[self.current_optimizer])
            trainer.trainModel()
            current_accuracy = trainer.run()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.current_mode = mode
        # find the best feature extractor
        for feature_extractor in range(1,len(self.feature_extractors)):
            trainer.createModel(
                self.modes[self.current_mode], 
                self.feature_extractors[feature_extractor], \
                self.encoders[self.current_encoder], \
                self.decoders[self.current_decoder], \
                self.encoder_sizes[self.current_encoder_size], \
                self.decoder_sizes[self.current_decoder_size], \
                self.optimizers[self.current_optimizer])
            current_accuracy = Trainer.run()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.current_feature_extractor = feature_extractor
        # find the best encoder
        for encoder in range(1,len(self.encoders)):
            trainer.createModel(
                self.modes[self.current_mode], 
                self.feature_extractors[self.current_feature_extractor], \
                self.encoders[encoder], \
                self.decoders[self.current_decoder], \
                self.encoder_sizes[self.current_encoder_size], \
                self.decoder_sizes[self.current_decoder_size], \
                self.optimizers[self.current_optimizer])
            current_accuracy = Trainer.run()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.current_encoder = encoder                
        # find the best decoder
        for decoder in range(1,len(self.decoders)):
            trainer.createModel(
                self.modes[self.current_mode], 
                self.feature_extractors[self.current_feature_extractor], \
                self.encoders[self.current_encoder], \
                self.decoders[decoder], \
                self.encoder_sizes[self.current_encoder_size], \
                self.decoder_sizes[self.current_decoder_size], \
                self.optimizers[self.current_optimizer])
            current_accuracy = Trainer.run()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.current_decoder = decoder
        # find the best encoder size
        for encoder_size in range(1,len(self.encoder_sizes)):
            trainer.createModel(
                self.modes[self.current_mode], 
                self.feature_extractors[self.current_feature_extractor], \
                self.encoders[self.current_encoder], \
                self.decoders[self.current_decoder], \
                self.encoder_sizes[encoder_size], \
                self.decoder_sizes[self.current_decoder_size], \
                self.optimizers[self.current_optimizer])
            current_accuracy = Trainer.run()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.current_encoder_size = encoder_size
        # find the best decoder size
        for decoder_size in range(1,len(self.decoder_sizes)):
            trainer.createModel(
                self.modes[self.current_mode], 
                self.feature_extractors[self.current_feature_extractor], \
                self.encoders[self.current_encoder], \
                self.decoders[self.current_decoder], \
                self.encoder_sizes[self.current_encoder_size], \
                self.decoder_sizes[decoder_size], \
                self.optimizers[self.current_optimizer])
            current_accuracy = Trainer.run()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.current_decoder_size = decoder_size
        # find the best optimizer
        for optimizer in range(l1,en(self.optimizers)):
            trainer.createModel(
                self.modes[self.current_mode], 
                self.feature_extractors[feature_extractor], \
                self.encoders[self.current_encoder], \
                self.decoders[self.current_decoder], \
                self.encoder_sizes[self.current_encoder_size], \
                self.decoder_sizes[self.current_decoder_size], \
                self.optimizers[optimizer])
            current_accuracy = Trainer.run()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.current_optimizer = optimizer


def main(args):
    print('enter main method')
    try:
        trainer = Trainer.Trainer(os.environ['BASE_MODEL_DIR'], os.environ['DATA_DIR'],\
            os.environ['TMP_DIR'], 1000000)
        sess = tf.Session()
        print('greedy hyper parameter main')
        code.interact(local=dict(globals(), **locals()))
        ght = GreedyHyperparamterTuner()
        trainer.trainModel()
        #ght.findBestConfiguration()
    except:
        '''print('something went wrong!')
        print(sys.exc_info())
        code.interact(local=dict(globals(), **locals()))'''

if __name__ == '__main__':
    main(sys.argv[1:])

'''
execfile('GreedyHyperParameterTuner.py')
trainer.trainModel()
print('current problem')
code.interact(local=dict(globals(), **locals()))
'''