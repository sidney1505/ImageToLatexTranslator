import tensorflow as tf
import tflearn.layers.conv
from FeatureExtractor import FeatureExtractor

class VGGFLevelFeatureExtractor(FeatureExtractor):
    def __init__(self, model):
        FeatureExtractor.__init__(self, model)
        fe_size = self.model.encoder_size # prev 64
        self.channels =  [fe_size, 2 * fe_size, 4 * fe_size, 8 * fe_size, 8 * fe_size]

    def createGraph(self):
        with tf.variable_scope(self.model.feature_extractor, reuse=None):
            print('create cnn')
            self.model.featureLevels = []

            with tf.variable_scope("layer1", reuse=None):
                w1 = tf.get_variable( \
                    'weight', \
                    [3,3,1,self.channels[0]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.conv2d(self.model.input, w1, strides=[1,1,1,1], \
                    padding='SAME')
                b1 = tf.get_variable('bias', [self.channels[0]], tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b1)            
                features = tf.nn.relu(features)

            with tf.variable_scope("layer2", reuse=None):
                w1 = tf.get_variable( \
                    'weight', \
                    [3,3,self.channels[0],self.channels[0]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w1, strides=[1,1,1,1], padding='SAME')
                b1 = tf.get_variable( \
                    'bias', \
                    [self.channels[0]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b1)            
                features = tf.nn.relu(features)

            features = tflearn.layers.conv.max_pool_2d(features, [2,2], strides=[2,2])
            features = tf.contrib.layers.batch_norm(features)

            with tf.variable_scope("layer3", reuse=None):
                w2 = tf.get_variable( \
                    'weight', \
                    [3,3,self.channels[0],self.channels[1]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w2, strides=[1,1,1,1], padding='SAME')
                b2 = tf.get_variable( \
                    'bias', \
                    [self.channels[1]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b2)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer4", reuse=None):
                w2 = tf.get_variable( \
                    'weight', \
                    [3,3,self.channels[1],self.channels[1]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w2, strides=[1,1,1,1], padding='SAME')
                b2 = tf.get_variable( \
                    'bias', \
                    [self.channels[1]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b2)
                features = tf.nn.relu(features)

            self.model.reduction_rate = [2,2]
            self.model.featureLevels.append(features)
            features = tflearn.layers.conv.max_pool_2d(features, [2,1], strides=[2,1])
            features = tf.contrib.layers.batch_norm(features)
            # self.model.featureLevels.append(features)

            with tf.variable_scope("layer5", reuse=None):
                w3 = tf.get_variable( \
                    'weight', \
                    [3,3,self.channels[1],self.channels[2]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable( \
                    'bias', \
                    [self.channels[2]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.nn.relu(features)

            with tf.variable_scope("layer6", reuse=None):
                w3 = tf.get_variable( \
                    'weight', \
                    [3,3,self.channels[2],self.channels[2]], \
                    tf.float32, \
                    tf.random_normal_initializer())
                features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
                b3 = tf.get_variable('bias', [self.channels[2]], tf.float32, tf.random_normal_initializer())
                features = tf.nn.bias_add(features, b3)
                features = tf.nn.relu(features)

            features = tflearn.layers.conv.max_pool_2d(features, [1,2], strides=[1,2])            
            features = tf.contrib.layers.batch_norm(features)
            self.model.featureLevels.append(features)            
            self.model.featureLevels = self.model.featureLevels[::-1]

            return features