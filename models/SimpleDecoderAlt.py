import code
import tensorflow as tf

# https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq
def createGraph(model, features, state):
    statesize = state[0].shape[-1].value
    state = tf.contrib.rnn.LSTMStateTuple(state[0],state[1])
    batchsize = tf.shape(features)[0]
    model.decoder = "simpleDecoder"
    with tf.variable_scope(model.decoder, reuse=None):
        rnncell = tf.contrib.rnn.BasicLSTMCell(statesize)
        initial_state = state
        label_gold = tf.one_hot(model.label_gold, model.num_classes)
        def embedding(x):
            return tf.one_hot(x, model.num_classes)
        '''trainhelper = tf.contrib.seq2seq.TrainingHelper(label_gold, tf.zeros( \
            model.max_num_tokens, dtype=tf.int32) + model.max_num_tokens)'''
        #print('in simple decoder')
        #code.interact(local=dict(globals(), **locals()))
        output_embed = layers.embed_sequence(
            label_gold, vocab_size=len(model.vocabulary), embed_dim=256, scope='embed', reuse=True)
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')

        trainhelper = tf.contrib.seq2seq.TrainingHelper(output_embed, tf.zeros( \
            [batchsize], dtype=tf.int32) + model.max_num_tokens)
        start_tokens=tf.tile([0], [batchsize])
        end_token=model.num_classes - 1
        inferhelper = tf.contrib.seq2seq.SampleEmbeddingHelper(embeddings,start_tokens,end_token)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, trainhelper, initial_state)
        infer_decoder = tf.contrib.seq2seq.BasicDecoder(rnncell, inferhelper, initial_state)
        #helper = tf.cond(model.is_training, lambda: trainhelper, lambda: inferhelper)
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
