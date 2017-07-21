import tensorflow as tf
import tflearn.layers.conv

def createCNNModel(model, inputpx, loaded):
    with tf.variable_scope("wysiwygConv", reuse=None):
        print('create cnn')

        with tf.variable_scope("layer1", reuse=None):
            w1 = tf.get_variable('weight', [3,3,1,64], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(inputpx, w1, strides=[1,1,1,1], padding='SAME')
            b1 = tf.get_variable('bias', [64], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b1)            
            features = tf.nn.relu(features)

        with tf.variable_scope("layer2", reuse=None):
            w1 = tf.get_variable('weight', [3,3,1,64], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(inputpx, w1, strides=[1,1,1,1], padding='SAME')
            b1 = tf.get_variable('bias', [64], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b1)            
            features = tf.nn.relu(features)

        features = tflearn.layers.conv.max_pool_2d(features, [2,2], strides=[2,2])

        with tf.variable_scope("layer3", reuse=None):
            w2 = tf.get_variable('weight',[3,3,64,128], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w2, strides=[1,1,1,1], padding='SAME')
            b2 = tf.get_variable('bias', [128], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b2)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer4", reuse=None):
            w2 = tf.get_variable('weight',[3,3,64,128], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w2, strides=[1,1,1,1], padding='SAME')
            b2 = tf.get_variable('bias', [128], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b2)
            features = tf.nn.relu(features)

        features = tflearn.layers.conv.max_pool_2d(features, [2,1], strides=[2,1])

        with tf.variable_scope("layer5", reuse=None):
            w3 = tf.get_variable('weight',[3,3,128,256], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
            b3 = tf.get_variable('bias', [256], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b3)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer6", reuse=None):
            w3 = tf.get_variable('weight',[3,3,128,256], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
            b3 = tf.get_variable('bias', [256], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b3)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer7", reuse=None):
            w3 = tf.get_variable('weight',[3,3,128,256], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
            b3 = tf.get_variable('bias', [256], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b3)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer8", reuse=None):
            w3 = tf.get_variable('weight',[3,3,128,256], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w3, strides=[1,1,1,1], padding='SAME')
            b3 = tf.get_variable('bias', [256], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b3)
            features = tf.nn.relu(features)

        features = tflearn.layers.conv.max_pool_2d(features, [1,2], strides=[1,2])            
        # features = tf.contrib.layers.batch_norm(features)

        with tf.variable_scope("layer9", reuse=None):
            w5 = tf.get_variable('weight',[3,3,256,512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w5, strides=[1,1,1,1], padding='SAME')
            b5 = tf.get_variable('bias', [512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b5)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer10", reuse=None):
            w5 = tf.get_variable('weight',[3,3,256,512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w5, strides=[1,1,1,1], padding='SAME')
            b5 = tf.get_variable('bias', [512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b5)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer11", reuse=None):
            w5 = tf.get_variable('weight',[3,3,256,512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w5, strides=[1,1,1,1], padding='SAME')
            b5 = tf.get_variable('bias', [512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b5)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer12", reuse=None):
            w5 = tf.get_variable('weight',[3,3,256,512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w5, strides=[1,1,1,1], padding='SAME')
            b5 = tf.get_variable('bias', [512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b5)
            features = tf.nn.relu(features)
        
        features = tflearn.layers.conv.max_pool_2d(features, [2,1], strides=[2,1])

        with tf.variable_scope("layer13", reuse=None):
            w6 = tf.get_variable('weight',[3,3,512,512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w6, strides=[1,1,1,1], padding='SAME')
            b6 = tf.get_variable('bias', [512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b6)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer14", reuse=None):
            w6 = tf.get_variable('weight',[3,3,512,512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w6, strides=[1,1,1,1], padding='SAME')
            b6 = tf.get_variable('bias', [512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b6)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer15", reuse=None):
            w6 = tf.get_variable('weight',[3,3,512,512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w6, strides=[1,1,1,1], padding='SAME')
            b6 = tf.get_variable('bias', [512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b6)
            features = tf.nn.relu(features)

        with tf.variable_scope("layer16", reuse=None):
            w6 = tf.get_variable('weight',[3,3,512,512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.conv2d(features, w6, strides=[1,1,1,1], padding='SAME')
            b6 = tf.get_variable('bias', [512], tf.float32, tf.random_normal_initializer())
            features = tf.nn.bias_add(features, b6)
            features = tf.nn.relu(features)

        features = tflearn.layers.conv.max_pool_2d(features, [1,2], strides=[1,2])

        return features