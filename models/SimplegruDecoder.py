import code
import tensorflow as tf

# https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq
def createGraph(model, features, state):
    state = tf.concat([state[0],state[1]],1)
    statesize = state.shape[-1].value
    batchsize = tf.shape(features)[0]
    model.decoder = "simpleDecoder"
    with tf.variable_scope(model.decoder, reuse=None):
        rnncell = tf.contrib.rnn.GRUCell(statesize)
        #code.interact(local=dict(globals(), **locals()))
        initial_state = state
        label_gold = tf.one_hot(model.label_gold, model.num_classes)
        helper = tf.contrib.seq2seq.TrainingHelper(label_gold, tf.zeros( \
            model.max_num_tokens, dtype=tf.int32) + model.max_num_tokens)
        decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, helper, initial_state)
        embedding=lambda x: tf.one_hot(x, model.num_classes)
        start_tokens=tf.tile([0], [batchsize])
        end_token=model.num_classes - 1
        inferhelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding,start_tokens,end_token)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, trainhelper, initial_state)
        infer_decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, inferhelper, initial_state)
        # code.interact(local=dict(globals(), **locals()))
        final_outputs, final_state, final_sequence_lengths = tf.cond(model.is_training, \
            lambda: tf.contrib.seq2seq.dynamic_decode(train_decoder, \
                maximum_iterations=model.max_num_tokens), \
            lambda: tf.contrib.seq2seq.dynamic_decode(infer_decoder,\
                maximum_iterations=model.max_num_tokens))
        outputs = final_outputs[0]
        w1 = tf.get_variable('w1', [statesize,model.num_classes], tf.float32, \
            tf.random_normal_initializer())
        prediction = tf.tensordot(outputs,w1,[[2],[0]])
        b1 = tf.get_variable('b1', [model.num_classes], tf.float32, \
            tf.random_normal_initializer())
        prediction = prediction + b1
    return prediction
