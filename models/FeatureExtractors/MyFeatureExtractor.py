import tensorflow as tf
import tflearn.layers.conv
from FeatureExtractor import FeatureExtractor

class VGGFinegrainedFeatureExtractor(FeatureExtractor):
    def __init__(self, model):
        FeatureExtractor.__init__(self, model)

    def createGraph(self):
        with tf.variable_scope(self.model.feature_extractor, reuse=None):
            print('create cnn')

            size_a = self.model.feature_extractor_size / 2

            with tf.variable_scope("layer1", reuse=None):
                w1 = tf.get_variable('weight', [3,3,1,size_a], tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.conv2d(self.model.input, w1, strides=[1,1,1,1], \
                    padding='SAME')
                b1 = tf.get_variable('bias', [size_a], tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b1)            
                features = tf.nn.relu(features)

            with tf.variable_scope("layer2", reuse=None):
                w1 = tf.get_variable('weight', [3,3,size_a,size_a], tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w1, strides=[1,1,1,1], padding='SAME')
                b1 = tf.get_variable('bias', [size_a], tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b1)            
                features = tf.nn.relu(features)

            with tf.variable_scope("layer3", reuse=None):
                w2 = tf.get_variable('weight',[3,3,size_a,size_a], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w2, strides=[1,1,1,1], padding='SAME')
                b2 = tf.get_variable('bias', [size_a], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b2)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer4", reuse=None):
                w2 = tf.get_variable('weight',[3,3,size_a,size_a], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w2, strides=[1,1,1,1], padding='SAME')
                b2 = tf.get_variable('bias', [size_a], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b2)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer5", reuse=None):
                w3 = tf.get_variable('weight',[3,3,size_a,size_a], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable('bias', [256], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.nn.relu(features)

            features = tflearn.layers.conv.max_pool_2d(features, [2,2], strides=[2,2])
            features = tf.contrib.layers.batch_norm(features)

            size_b = self.model.feature_extractor_size / 2

            with tf.variable_scope("layer6", reuse=None):
                w3 = tf.get_variable('weight',[3,3,size_a,size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable('bias', [size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer7", reuse=None):
                w3 = tf.get_variable('weight',[3,3,size_b,size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable('bias', [size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer8", reuse=None):
                w3 = tf.get_variable('weight',[3,3,size_b,size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable('bias', [size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer9", reuse=None):
                w3 = tf.get_variable('weight',[3,3,size_b,size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable('bias', [size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer10", reuse=None):
                w3 = tf.get_variable('weight',[3,3,size_b,size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable('bias', [size_b], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.nn.relu(features)

            features = tflearn.layers.conv.max_pool_2d(features, [2,2], strides=[2,2])
            features = tf.contrib.layers.batch_norm(features)

            return features