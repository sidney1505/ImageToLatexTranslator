import code
import tensorflow as tf
import tflearn

def createGraph(model, features):
    #code.interact(local=dict(globals(), **locals()))
    with tf.variable_scope("vggClassifier", reuse=None):
        features = tflearn.layers.conv.global_avg_pool(features)
        with tf.variable_scope("layer1", reuse=None):
            w1 = tf.get_variable('weight', [features.shape[1],4096], tf.float32, \
                tf.random_normal_initializer())
            features = tf.tensordot(features,w1,[[1],[0]])
            b1 = tf.get_variable('bias', [4096], tf.float32, tf.random_normal_initializer())
            features = features + b1
            features = tf.nn.relu(features)
            features = tf.nn.dropout(features,model.keep_prob)

        with tf.variable_scope("layer2", reuse=None):
            w1 = tf.get_variable('weight', [4096,4096],tf.float32, \
                tf.random_normal_initializer())
            features = tf.tensordot(features,w1,[[1],[0]])
            b1 = tf.get_variable('bias', [4096], tf.float32, tf.random_normal_initializer())
            features = features + b1
            features = tf.nn.relu(features)
            features = tf.nn.dropout(features,model.keep_prob)

        with tf.variable_scope("layer3", reuse=None):
            w1 = tf.get_variable('weight', [4096,model.num_classes],tf.float32, \
                tf.random_normal_initializer())
            features = tf.tensordot(features,w1,[[1],[0]])
            b1 = tf.get_variable('bias', [model.num_classes], tf.float32, \
                tf.random_normal_initializer())
            features = features + b1

    return features
