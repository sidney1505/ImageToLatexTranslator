import tensorflow as tf
import tflearn.layers.conv

class FeatureExtractor():
    def __init__(self, model):
        self.model = model

    def createGraph(self):
        raise NotImplementedError
