import tensorflow as tf

def createBaselineDecoder(model,state):
    with tf.variable_scope("baseLineDecoder", reuse=None):
        model.weights.update({
            'wfcbl': tf.Variable(tf.random_normal([2048,model.num_classes]), name='wfcbl'),
            'bfcbl': tf.Variable(tf.random_normal([model.num_classes]), name='bfcbl')
        })
        #
        initial_in = tf.random_normal(tf.shape(state[0]))
        rnncell = tf.contrib.rnn.LSTMCell(2048)
        outputs = tf.TensorArray(dtype=tf.float32, size=model.max_num_tokens)
        state = tf.contrib.rnn.LSTMStateTuple(state[0],state[1])
        params = [tf.constant(0), initial_in, outputs, state]
        while_condition = lambda i, inp, outputs, state: tf.less(i, model.max_num_tokens)
        def body(i, inp, outputs, state):
            output, state = rnncell.__call__(inp, state)
            # code.interact(local=locals())
            outputs = outputs.write(i, output)
            return [tf.add(i, 1), output, outputs, state]
        _,_,outputs,_ = tf.while_loop(while_condition, body, params)
        outputs = outputs.stack()
        prediction = tf.tensordot(outputs,model.weights['wfcbl'],[[2],[0]]) 
        prediction = prediction + model.weights['bfcbl']
        return tf.nn.softmax(prediction)