import code
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

# https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq
def createGraph(model, features, state):
    #code.interact(local=dict(globals(), **locals()))
    statesize = state[0].shape[-1].value # * 2
    c = state[0] # tf.concat([state[0],state[0]],1)
    h = state[1] # tf.concat([state[1],state[1]],1)
    state = tf.contrib.rnn.LSTMStateTuple(c,h)
    batchsize = tf.shape(features)[0]
    model.decoder = "simpleDecoder"
    with tf.variable_scope(model.decoder, reuse=None):
        rnncell = tf.contrib.rnn.BasicLSTMCell(statesize)
        initial_state = state
        GO_SYMBOL = -1
        END_SYMBOL = model.num_classes
        start_tokens=tf.tile([GO_SYMBOL], [batchsize])
        start_tokens2D = tf.expand_dims(start_tokens, 1)
        decoder_input = tf.concat([start_tokens2D, model.label_gold], 1)
        label_gold = tf.one_hot(decoder_input, model.num_classes)
        def embedding(x):
            return tf.one_hot(x, model.num_classes)
        '''trainhelper = tf.contrib.seq2seq.TrainingHelper(label_gold, tf.zeros( \
            model.max_num_tokens, dtype=tf.int32) + model.max_num_tokens)'''
        trainhelper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(label_gold, \
            tf.zeros([batchsize], dtype=tf.int32) + model.max_num_tokens, embedding, 0.0)
        inferhelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding,start_tokens,\
            END_SYMBOL)
        projection_layer = layers_core.Dense(model.num_classes, use_bias=False)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, trainhelper, \
            initial_state, output_layer=projection_layer)
        infer_decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, inferhelper, \
            initial_state, output_layer=projection_layer)
        #infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(rnncell,embedding,\
        #    start_tokens,END_SYMBOL,initial_state,beam_width=5, \
        #    output_layer=projection_layer)
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
        '''w1 = tf.get_variable('w1', [statesize,model.num_classes], tf.float32, \
            tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [model.num_classes], tf.float32, \
            tf.random_normal_initializer())'''
        model.train_prediction = train_final_outputs[0]
        '''train_prediction = tf.tensordot(train_outputs,w1,[[2],[0]])
        model.train_prediction = train_prediction + b1'''
        model.infer_prediction = infer_final_outputs[0]
        '''infer_prediction = tf.tensordot(infer_outputs,w1,[[2],[0]])
        model.infer_prediction = infer_prediction + b1'''
        #code.interact(local=dict(globals(), **locals()))
    return tf.cond(model.is_training, lambda:model.train_prediction, \
        lambda:model.infer_prediction)
