import Trainer

class GreedyHyperparamterTuner:
    def __init__(self):
        self.modes=['e2e','stepbystep']
        self.feature_extractors=['alexnetFe','wysiwysgFe','vggFe','resnetFe','densenetFe']
        self.encoders=['simpleEnc','monorowEnc','birowEnc','monocolEnc','bicolEnc','quadroEnc']
        self.decoders=['simpleDec','simplegruDec','bahdanauDec','luongDec']
        self.encoder_sizes=[512,1024,2048,4096]
        self.decoder_sizes=[512,1024,2048,4096]
        self.optimizers=['vanillaSGD','momentum','adam','adadelta']

    def findBestConfiguration():
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
            current_accuracy = Trainer.run()
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
    trainer = Trainer.Trainer(os.environ['BASE_MODEL_DIR'], os.environ['DATA_DIR'],\
        os.environ['TMP_DIR'])
    ght = GreedyHyperparamterTuner()
    code.interact(local=dict(globals(), **locals()))
    ght.findBestConfiguration()

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')