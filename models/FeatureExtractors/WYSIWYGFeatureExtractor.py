import tensorflow as tf
import tflearn.layers.conv
from FeatureExtractor import FeatureExtractor

class WYSIWYGFeatureExtractor(FeatureExtractor):
    def __init__(self, model):
        FeatureExtractor.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope(self.model.feature_extractor, reuse=None):
            with tf.variable_scope("layer1", reuse=None):
                w1 = tf.get_variable('weight', [3,3,1,64], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(self.model.input, w1, strides=[1,1,1,1], padding='SAME')
                b1 = tf.get_variable('bias', [64], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b1)
                features = tflearn.layers.conv.max_pool_2d(features, 2, strides=2)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer2", reuse=None):
                w2 = tf.get_variable('weight',[3,3,64,128], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w2, strides=[1,1,1,1], padding='SAME')
                b2 = tf.get_variable('bias', [128], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b2)
                features = tflearn.layers.conv.max_pool_2d(features, 2, strides=2)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer3", reuse=None):
                w3 = tf.get_variable('weight',[3,3,128,256], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable('bias', [256], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.contrib.layers.batch_norm(features)
                #features = tflearn.layers.normalization.batch_normalization(features) #same as torch?
                features = tf.nn.relu(features)

            with tf.variable_scope("layer4", reuse=None):
                w4 = tf.get_variable('weight',[3,3,256,256], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w4, strides=[1,1,1,1], padding='SAME')
                b4 = tf.get_variable('bias', [256], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b4)
                features = tflearn.layers.conv.max_pool_2d(features, [2,1], strides=[2,1])
                features = tf.nn.relu(features)

            with tf.variable_scope("layer5", reuse=None):
                w5 = tf.get_variable('weight',[3,3,256,512], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w5, strides=[1,1,1,1], padding='SAME')
                b5 = tf.get_variable('bias', [512], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b5)
                features = tf.contrib.layers.batch_norm(features)
                #features = tflearn.layers.normalization.batch_normalization(features) #same as torch?
                features = tflearn.layers.conv.max_pool_2d(features, [1,2], strides=[1,2])
                features = tf.nn.relu(features)

            with tf.variable_scope("layer6", reuse=None):
                w6 = tf.get_variable('weight',[3,3,512,self.model.num_features], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w6, strides=[1,1,1,1], padding='SAME')
                b6 = tf.get_variable('bias', [self.model.num_features], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b6)
                features = tf.contrib.layers.batch_norm(features)
                #features = tflearn.layers.normalization.batch_normalization(features)
                features = tf.nn.relu(features)

            return features