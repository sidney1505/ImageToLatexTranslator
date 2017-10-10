import code
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers import core as layers_core


# https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq
def createGraph(model, features, state):
    statesize = state[0].shape[-1].value
    state = tf.contrib.rnn.LSTMStateTuple(state[0],state[1])
    batchsize = tf.shape(features)[0]
    model.decoder = "simpleDecoder"
    with tf.variable_scope(model.decoder, reuse=None):
        rnncell = tf.contrib.rnn.BasicLSTMCell(statesize)
        initial_state = state
        GO_SYMBOL=model.num_classes
        END_SYMBOL=model.num_classes - 1
        #label_gold = tf.one_hot(model.label_gold, model.num_classes)
        '''def embedding(x):
            return tf.one_hot(x, model.num_classes)
        trainhelper = tf.contrib.seq2seq.TrainingHelper(label_gold, tf.zeros( \
            model.max_num_tokens, dtype=tf.int32) + model.max_num_tokens)'''
        start_tokens=tf.tile([GO_SYMBOL], [batchsize])
        # code.interact(local=dict(globals(), **locals()))
        #start_tokenss = tf.expand_dims(start_tokens, 1)
        #decoder_input = tf.concat([start_tokenss, model.label_gold], 1)
        decoder_input = model.label_gold
        output_embed = layers.embed_sequence(decoder_input, vocab_size= \
            len(model.vocabulary) + 1, embed_dim=256, scope='embed')
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')

        trainhelper = tf.contrib.seq2seq.TrainingHelper(output_embed, tf.zeros( \
            [batchsize], dtype=tf.int32) + model.max_num_tokens)
        inferhelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding,start_tokens,\
            END_SYMBOL)
        projection_layer = layers_core.Dense(model.num_classes, use_bias=False)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, trainhelper, \
            initial_state, output_layer=projection_layer)
        infer_decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, inferhelper, \
            initial_state, output_layer=projection_layer)        
        #helper = tf.cond(model.is_training, lambda: trainhelper, lambda: inferhelper)
        '''final_outputs, final_state, final_sequence_lengths = tf.cond(model.is_training, \
            lambda: tf.contrib.seq2seq.dynamic_decode(train_decoder, \
                maximum_iterations=model.max_num_tokens), \
            lambda: tf.contrib.seq2seq.dynamic_decode(infer_decoder,\
                maximum_iterations=model.max_num_tokens))'''
        train_final_outputs, train_final_state, train_final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(train_decoder,\
            maximum_iterations=model.max_num_tokens)
        infer_final_outputs, infer_final_state2, infer_final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(infer_decoder, \
            maximum_iterations=model.max_num_tokens)
        model.train_prediction = train_final_outputs[0]
        model.infer_prediction = infer_final_outputs[0]
    return prediction
