
def main(args):
    print('enter main method')
    trainer = Trainer.Trainer(os.environ['BASE_MODEL_DIR'], os.environ['DATA_DIR'],\
        os.environ['MONICA_TMP_DIR'], 1000000)
    # trainer.setModelParameters(self, mode, feature_extractor, encoder, decoder, encoder_size, \
    #        decoder_size, optimizer)
    trainer.trainModel()

if __name__ == '__main__':
    main(sys.argv[1:])