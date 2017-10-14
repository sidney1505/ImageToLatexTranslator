import tensorflow as tf
import tflearn.layers.conv
from FeatureExtractor import FeatureExtractor

class AlexnetFeatureExtractor(FeatureExtractor):
    def __init__(self, model):
        FeatureExtractor.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope("alexnetFe", reuse=None):
            with tf.variable_scope("layer1", reuse=None):
                w1 = tf.get_variable('weight', [11,11,1,96], tf.float32, \
                	tf.random_normal_initializer())
                features = tf.nn.conv2d(self.model.input, w1, strides=[1,1,1,1], \
                    padding='SAME')
                b1 = tf.get_variable('bias', [96], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b1)            
                features = tf.nn.relu(features)

            features = tf.contrib.layers.batch_norm(features)
            features = tflearn.layers.conv.max_pool_2d(features, [2,2], strides=[2,2])

            with tf.variable_scope("layer2", reuse=None):
                w1 = tf.get_variable('weight', [11,11,96,256], tf.float32, \
                	tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w1, strides=[1,1,1,1], padding='SAME')
                b1 = tf.get_variable('bias', [256], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b1)            
                features = tf.nn.relu(features)

            features = tf.contrib.layers.batch_norm(features)
            features = tflearn.layers.conv.max_pool_2d(features, [2,2], strides=[2,2])

            with tf.variable_scope("layer3", reuse=None):
                w2 = tf.get_variable('weight',[3,3,256,384], tf.float32, \
                	tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w2, strides=[1,1,1,1], padding='SAME')
                b2 = tf.get_variable('bias', [384], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b2)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer4", reuse=None):
                w2 = tf.get_variable('weight',[3,3,384,384], tf.float32, \
                	tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w2, strides=[1,1,1,1], padding='SAME')
                b2 = tf.get_variable('bias', [384], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b2)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer5", reuse=None):
                w3 = tf.get_variable('weight',[3,3,384,self.model.num_features], tf.float32,\
                	tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable('bias', [self.model.num_features], tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.nn.relu(features)
                 
            return features