import code
import os
import shutil
import tensorflow as tf
import tflearn

# https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq
def createAttentionModule(model, features, state):
    with tf.variable_scope("fullyConnectedVGG", reuse=None):
        features = tf.reshape(features,[tf.shape[0],tf.shape[1]*tf.shape[2],features.shape[3]])
        rnncell = tf.contrib.rnn.BasicLSTMCell(1024)
        attention = tf.contrib.seq2seq.BahdanauAttention(1024, features)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(rnncell, attention)
        helper = tf.contrib.seq2seq.TrainingHelper(features, model.max_num_tokens)
        decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, state)
        final_outputs, final_state, final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(decoder)
    return final_outputs
