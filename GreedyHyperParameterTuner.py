import sys, os, code, shutil
import Trainer
import tensorflow as tf

class GreedyHyperparamterTuner:
    def __init__(self, initialize=False, best_param_dir=None):
        self.modes=['e2e','stepbystep']
        self.feature_extractors=['wysiwysgFe','alexnetFe','vggFe','resnetFe']
        self.encoders=['monorowEnc','birowEnc','monocolEnc','bicolEnc','quadroEnc']
        self.decoders=['simpleDec','simplegruDec','bahdanauDec','luongDec']
        self.encoder_sizes=[1024,2048,4096]
        self.decoder_sizes=[1024,2048,4096]
        self.optimizers=['sgd','momentum','adam','adadelta']
        #
        self.trainer = Trainer.Trainer(os.environ['BASE_MODEL_DIR'], os.environ['DATA_DIR'],\
            os.environ['TMP_DIR'], 1000000)
        if best_param_dir != None:
            self.best_param_dir = best_param_dir
        else:
            self.best_param_dir = os.environ['BEST_MODEL_DIR']
        if initialize:
            self.initializeGreedySearch()
        self.current_mode = self.readParam('mode')
        self.current_feature_extractor = self.readParam('feature_extractor')
        self.current_encoder = self.readParam('encoder')
        self.current_decoder = self.readParam('decoder')
        self.current_encoder_size = int(self.readParam('encoder_size'))
        self.current_decoder_size = int(self.readParam('decoder_size'))
        self.current_optimizer = self.readParam('optimizer')
        self.current_best_accuracy = float(self.readParam('best_accuracy'))

    def initializeGreedySearch(self):
        self.trainer.setModelParameters('stepbystep', 'alexnetFe', 'monorowEnc', 'simpleDec',\
            1024, 1024, 'sgd')
        self.writeParam('mode', 'step_by_step')
        self.writeParam('feature_extractor', 'alexnetFe')
        self.writeParam('encoder', 'monorowEnc')
        self.writeParam('decoder', 'simpleDec')
        self.writeParam('encoder_size', 1024)
        self.writeParam('decoder_size', 1024)
        self.writeParam('optimizer', 'sgd')
        self.trainer.trainModel()
        accuracy = self.trainer.testModel()
        self.writeParam('best_accuracy', accuracy)

    def readParam(self, param):
        read_path = self.best_param_dir + '/' + param
        if not os.path.exists(read_path):
            raise Exception(read_path + ' does not exist!')
        reader = open(read_path, 'r')
        value = reader.read()
        reader.close()
        return value

    def writeParam(self, param, value):
        if not os.path.exists(self.best_param_dir):
            os.makedirs(self.best_param_dir)
        write_path = self.best_param_dir + '/' + param
        shutil.rmtree(write_path, ignore_errors=True)
        writer = open(write_path, 'w')
        writer.write(str(value))
        writer.close()

    def testMode(self, mode):
        model = self.trainer.__findBestModel(mode, self.current_feature_extractor, \
            self.current_encoder, self.current_decoder, self.current_encoder_size, \
            self.current_decoder_size, self.current_optimizer)
        if model != None:
            self.trainer.loadModel(model)
        else:
            self.trainer.setModelParameters(mode, self.current_feature_extractor, \
                self.current_encoder, self.current_decoder, self.current_encoder_size, \
                self.current_decoder_size, self.current_optimizer)
            self.trainer.trainModel()
        model_accuracy = trainer.testModel()
        if self.current_best_accuracy < model_accuracy:
            self.writeParam('best_accuracy', model_accuracy)
            self.writeParam('mode', mode)

    def testFeatureExtractor(self, mode):
        model = self.trainer.__findBestModel(mode, self.current_feature_extractor, \
            self.current_encoder, self.current_decoder, self.current_encoder_size, \
            self.current_decoder_size, self.current_optimizer)
        if model != None:
            self.trainer.loadModel(model)
        else:
            self.trainer.setModelParameters(mode, self.current_feature_extractor, \
                self.current_encoder, self.current_decoder, self.current_encoder_size, \
                self.current_decoder_size, self.current_optimizer)
            self.trainer.trainModel()
        model_accuracy = trainer.testModel()
        if self.current_best_accuracy < model_accuracy:
            self.writeParam('best_accuracy', model_accuracy)
            self.writeParam('mode', mode)


def main(args):
    print('enter main method')
    trainer = Trainer.Trainer(os.environ['BASE_MODEL_DIR'], os.environ['DATA_DIR'],\
        os.environ['TMP_DIR'], 250000)
    #sess = tf.Session()
    #print('greedy hyper parameter main')
    ght = GreedyHyperparamterTuner()
    #trainer.trainModel()
    code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main(sys.argv[1:])

'''
execfile('GreedyHyperParameterTuner.py')
trainer.trainModel()
print('current problem')
code.interact(local=dict(globals(), **locals()))
'''