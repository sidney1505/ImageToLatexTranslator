import code
import tensorflow as tf

# https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq
def createGraph(model, features, state):
    statesize = state.shape[-1]
    state = tf.contrib.rnn.LSTMStateTuple(state[0],state[1])
    model.decoder = "simpleDecoder"
    with tf.variable_scope(model.decoder, reuse=None):
        rnncell = tf.contrib.rnn.BasicLSTMCell(statesize)
        initial_state = rnncell.zero_state(tf.shape(features)[0],tf.float32).clone( \
            cell_state=state)
        label_gold = tf.one_hot(model.label_gold, model.num_classes)
        helper = tf.contrib.seq2seq.TrainingHelper(label_gold, tf.zeros( \
            model.max_num_tokens, dtype=tf.int32) + model.max_num_tokens)
        decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, helper, initial_state)
        final_outputs, final_state, final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=model.max_num_tokens)
        outputs = final_outputs[0]
        #code.interact(local=dict(globals(), **locals()))
        w1 = tf.get_variable('w1', [statesize,model.num_classes], tf.float32, \
            tf.random_normal_initializer())
        prediction = tf.tensordot(outputs,w1,[[2],[0]])
        b1 = tf.get_variable('b1', [model.num_classes], tf.float32, \
            tf.random_normal_initializer())
        prediction = prediction + b1
    return prediction
