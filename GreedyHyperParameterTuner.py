import Trainer

class GreedyHyperparamterTuner:
	def __init__(self):
		self.modes=['e2e','stepbystep']
		self.feature_extractors=['alexnetFe','wysiwysgFe','vggFe','resnetFe','densenetFe']
		self.encoders=['simpleEnc','monorowEnc','birowEnc','monocolEnc','bicolEnc','quadroEnc']
		self.decoders=['simpleDec','simplegruDec','bahdanauDec','luongDec']
		self.encoder_size=[512,1024,2048,4096]
		self.decoder_size=[512,1024,2048,4096]
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
		for mode in self.modes:
			current_trainer = Trainer.Trainer(
				self.modes[mode], 
				self.feature_extractors[self.current_feature_extractor], \
				self.encoders[self.current_encoder], \
				self.decoders[self.current_decoder], \
				self.encoder_size[self.current_encoder_size], \
				self.decoder_size[self.current_decoder_size], \
				self.optimizers[self.current_optimizer])
			current_accuracy = Trainer.run()
			if current_accuracy > best_accuracy:
				best_accuracy = current_accuracy
				current_mode = mode
		# find the best feature extractor



def main(args):
    ght = GreedyHyperparamterTuner()
    ght.findBestConfiguration()

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')