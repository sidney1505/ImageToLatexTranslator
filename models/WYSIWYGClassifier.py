import code
import tensorflow as tf
import tflearn

def createGraph(model, features):
    model.classifier = "fullyConnected"
    with tf.variable_scope(model.classifier, reuse=None):
        features = tflearn.layers.conv.global_avg_pool(features)

        w1 = tf.get_variable('w1', [features.shape[1],1000], tf.float32, \
            tf.random_normal_initializer())
        features = tf.tensordot(features,w1,[[1],[0]])
        b1 = tf.get_variable('b1', [1000], tf.float32, tf.random_normal_initializer())
        features = features + b1
        features = tf.nn.relu(features)

        w2 = tf.get_variable('w2', [1000,model.num_classes], tf.float32, \
            tf.random_normal_initializer())
        features = tf.tensordot(features,w2,[[1],[0]])
        b2 = tf.get_variable('b2', [model.num_classes], tf.float32, \
            tf.random_normal_initializer())
        features = features + b2

        return features
