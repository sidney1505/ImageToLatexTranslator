import tensorflow as tf
import tflearn.layers.conv
from FeatureExtractor import FeatureExtractor

class ResnetFeatureExtractor(FeatureExtractor):
    def __init__(self, model):
        FeatureExtractor.__init__(self, model)

    def createGraph(self):
        # Residual blocks
        # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
        n = 5
        with tf.variable_scope(self.model.feature_extractor, reuse=None):
            net = tflearn.conv_2d(self.model.input, 16, 3, regularizer='L2', \
                weight_decay=0.0001)
            net = tflearn.residual_block(net, n, 16)
            net = tflearn.residual_block(net, 1, 32, downsample=True)
            net = tflearn.residual_block(net, n-1, 32)
            net = tflearn.residual_block(net, 1, 64, downsample=True)
            net = tflearn.residual_block(net, n-1, 64)
            net = tflearn.batch_normalization(net)
            net = tflearn.activation(net, 'relu')
            return net