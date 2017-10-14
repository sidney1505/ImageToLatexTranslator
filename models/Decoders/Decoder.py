import code
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

class Decoder():
    def __init__(self, model):
        self.model = model

    def createGraph(self):
        self.statesize = self.model.state[0].shape[-1].value # * 2
        c = self.model.state[0] # tf.concat([state[0],state[0]],1)
        h = self.model.state[1] # tf.concat([state[1],state[1]],1)
        state = tf.contrib.rnn.LSTMStateTuple(c,h)
        batchsize = tf.shape(self.model.features)[0]
        with tf.variable_scope(self.model.decoder, reuse=None):
            decodercell = self.createDecoderCell()
            initial_state = state
            GO_SYMBOL = -1
            END_SYMBOL = self.model.num_classes
            start_tokens=tf.tile([GO_SYMBOL], [batchsize])
            start_tokens2D = tf.expand_dims(start_tokens, 1)
            decoder_input = tf.concat([start_tokens2D, self.model.groundtruth], 1)
            label_gold = tf.one_hot(decoder_input, self.model.num_classes)
            def embedding(x):
                return tf.one_hot(x, self.model.num_classes)
            trainhelper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(label_gold, \
                tf.zeros([batchsize], dtype=tf.int32) + self.model.max_num_tokens, \
                embedding, 0.0)
            inferhelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding,start_tokens,\
                END_SYMBOL)
            projection_layer = layers_core.Dense(self.model.num_classes, use_bias=False)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(decodercell, \
                trainhelper, initial_state, output_layer=projection_layer)
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(decodercell, \
                inferhelper, initial_state, output_layer=projection_layer)
            train_final_outputs, train_final_state, train_final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(train_decoder,\
                maximum_iterations=self.model.max_num_tokens)
            infer_final_outputs, infer_final_state2, infer_final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(infer_decoder, \
                maximum_iterations=self.model.max_num_tokens)
            self.model.train_prediction = train_final_outputs[0]
            self.model.infer_prediction = infer_final_outputs[0]

    def createDecoderCell(self):
        raise NotImplementedError
