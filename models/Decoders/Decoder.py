import code
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

class Decoder():
    def __init__(self, model):
        self.model = model

    def createGraph(self):
        self.batchsize = tf.shape(self.model.refined_features)[0]
        with tf.variable_scope(self.model.decoder, reuse=None):
            #GO_SYMBOL = self.model.num_classes
            GO_SYMBOL = -1
            END_SYMBOL = self.model.num_classes - 1
            start_tokens=tf.tile([GO_SYMBOL], [self.batchsize])
            start_tokens2D = tf.expand_dims(start_tokens, 1)
            #src_vocab_size = self.model.num_classes + 1
            src_vocab_size = self.model.num_classes
            #src_embed_size = 256 # 256?
            #embedding = tf.get_variable("embedding", [src_vocab_size, \
            #    src_embed_size], dtype=tf.float32)            
            def embedding(x):
                return tf.one_hot(x, src_vocab_size)
            projection_layer = layers_core.Dense(self.model.num_classes, use_bias=False)

            if not self.model.only_inference:
                decodercell = self.createDecoderCell()
                # using the train helper
                nr_target = tf.argmax(self.model.groundtruth, axis=-1) + 1
                nr_target = tf.to_int32(nr_target)
                #nr_target = tf.zeros([self.batchsize], dtype=tf.int32) + self.model.max_num_tokens
                decoder_input = tf.concat([start_tokens2D, self.model.groundtruth], 1)
                #decoder_emb_inp = tf.nn.embedding_lookup(embedding, decoder_input)
                decoder_emb_inp = embedding(decoder_input)
                #trainhelper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, nr_target)
                trainhelper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(\
                    decoder_emb_inp, nr_target, embedding, 0.0)
                train_decoder = tf.contrib.seq2seq.BasicDecoder(decodercell, \
                    trainhelper, self.initial_state, output_layer=projection_layer)
                train_final_outputs, train_final_state, train_final_sequence_lengths = \
                    tf.contrib.seq2seq.dynamic_decode(train_decoder,\
                    maximum_iterations=self.model.max_num_tokens)
                self.model.train_energy = train_final_outputs[0]
                self.model.train_distribution = tf.nn.softmax(self.model.train_energy)
                self.model.train_prediction = tf.argmax(self.model.train_distribution, axis=-1)
                # using the infer helper
                inferhelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, \
                    start_tokens, END_SYMBOL)
                infer_decoder = tf.contrib.seq2seq.BasicDecoder(decodercell, \
                    inferhelper, self.initial_state, output_layer=projection_layer)
                #infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decodercell, embedding, \
                #    start_tokens, END_SYMBOL, initial_state, 5, projection_layer)
                infer_final_outputs, infer_final_state, infer_final_sequence_lengths = \
                    tf.contrib.seq2seq.dynamic_decode(infer_decoder, \
                    maximum_iterations=self.model.max_num_tokens)
                self.model.infer_energy = infer_final_outputs[0]
                self.model.infer_distribution = tf.nn.softmax(self.model.infer_energy)
                self.model.infer_prediction = tf.argmax(self.model.infer_distribution, axis=2)
            else:
                beamsearch = True
                beamwidth = 10
                if not beamsearch:
                    decodercell = self.createDecoderCell()
                    # using the infer helper
                    inferhelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, \
                        start_tokens, END_SYMBOL)
                    infer_decoder = tf.contrib.seq2seq.BasicDecoder(decodercell, \
                        inferhelper, self.initial_state, output_layer=projection_layer)
                    #infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decodercell, embedding, \
                    #    start_tokens, END_SYMBOL, initial_state, 5, projection_layer)
                    infer_final_outputs, infer_final_state, infer_final_sequence_lengths = \
                        tf.contrib.seq2seq.dynamic_decode(infer_decoder, \
                        maximum_iterations=self.model.max_num_tokens)
                    self.model.infer_energy = infer_final_outputs[0]
                    self.model.infer_distribution = tf.nn.softmax(self.model.infer_energy)
                    self.model.infer_prediction = tf.argmax(self.model.infer_distribution, axis=2)
                else:
                    self.batchsize = self.batchsize * beamwidth
                    self.model.refined_features = tf.contrib.seq2seq.tile_batch( \
                        self.model.refined_features, beamwidth)
                    self.model.input_summary = tf.contrib.seq2seq.tile_batch( \
                        self.model.input_summary, beamwidth)
                    decodercell = self.createDecoderCell()
                    infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decodercell, embedding, \
                        start_tokens, END_SYMBOL, self.initial_state, beamwidth, \
                        projection_layer)
                    infer_final_outputs, infer_final_state, infer_final_sequence_lengths = \
                        tf.contrib.seq2seq.dynamic_decode(infer_decoder, \
                        maximum_iterations=self.model.max_num_tokens)
                    self.model.top_k = tf.transpose(infer_final_outputs[0],[2,0,1])
                    self.model.infer_prediction = self.model.top_k[0]
                    # code.interact(local=dict(globals(), **locals()))


    def getFirstEndtokens(self, inp):
        l = []
        code.interact(local=dict(globals(), **locals()))
        for batch in range(inp.shape[0]):
            for token in range(inp.shape[1]):
                if inp[batch][token] == self.model.num_classes - 1:
                    l.append(token + 1)
        return l

    def createDecoderCell(self):
        raise NotImplementedError
