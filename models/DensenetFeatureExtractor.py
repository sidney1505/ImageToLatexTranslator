import tensorflow as tf
import tflearn

def densenet_block(incoming, nb_layers, growth, bottleneck=True,
                   downsample=True, downsample_strides=2, activation='relu',
                   batch_norm=True, dropout=False, dropout_keep_prob=0.5,
                   weights_init='variance_scaling', regularizer='L2',
                   weight_decay=0.0001, bias=True, bias_init='zeros',
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="DenseNetBlock"):
    """ DenseNet Block.
    A DenseNet block as described in DenseNet paper.
    Input:
        4-D Tensor [batch, height, width, in_channels].
    Output:
        4-D Tensor [batch, new height, new width, out_channels].
    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        growth: `int`. DenseNet 'growth': The number of convolutional
            filters of each convolution.
        bottleneck: `bool`. If True, add a 1x1 convolution before the 3x3 
            convolution to reduce the number of input features map.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        dropout: `bool`. If True, apply dropout. Use 'dropout_keep_prob' to 
            specify the keep probability.
        dropout_keep_prob: `float`. Keep probability parameter for dropout.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'ResNeXtBlock'.
    References:
        Densely Connected Convolutional Networks, G. Huang, Z. Liu, 
        K. Q. Weinberger, L. van der Maaten. 2016.
    Links:
        [https://arxiv.org/abs/1608.06993]
        (https://arxiv.org/abs/1608.06993)
    """
    densenet = incoming

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        for i in range(nb_layers):

            # Identity
            conn = densenet

            # 1x1 Conv layer of the bottleneck block
            if bottleneck:
                if batch_norm:
                    densenet = tflearn.batch_normalization(densenet)
                densenet = tflearn.activation(densenet, activation)
                densenet = tflearn.conv_2d(densenet, nb_filter=growth,
                                   filter_size=1,
                                   bias=bias,
                                   weights_init=weights_init,
                                   bias_init=bias_init,
                                   regularizer=regularizer,
                                   weight_decay=weight_decay,
                                   trainable=trainable,
                                   restore=restore)

            # 3x3 Conv layer
            if batch_norm:
                densenet = tflearn.batch_normalization(densenet)
            densenet = tflearn.activation(densenet, activation)
            densenet = tflearn.conv_2d(densenet, nb_filter=growth,
                               filter_size=3,
                               bias=bias,
                               weights_init=weights_init,
                               bias_init=bias_init,
                               regularizer=regularizer,
                               weight_decay=weight_decay,
                               trainable=trainable,
                               restore=restore)

            # Connections
            densenet = tf.concat([densenet, conn], 3)

        # 1x1 Transition Conv
        if batch_norm:
            densenet = tflearn.batch_normalization(densenet)
        densenet = tflearn.activation(densenet, activation)
        densenet = tflearn.conv_2d(densenet, nb_filter=growth,
                           filter_size=1,
                           bias=bias,
                           weights_init=weights_init,
                           bias_init=bias_init,
                           regularizer=regularizer,
                           weight_decay=weight_decay,
                           trainable=trainable,
                           restore=restore)
        if dropout:
            densenet = tflearn.dropout(densenet, keep_prob=dropout_keep_prob)

        # Downsampling
        if downsample:
            densenet = tflearn.avg_pool_2d(densenet, kernel_size=2,
                                           strides=downsample_strides)

    return densenet

def createGraph(model, net):
    # Growth Rate (12, 16, 32, ...)
    k = 12

    # Depth (40, 100, ...)
    L = 40
    nb_layers = int((L - 4) / 3)
    with tf.variable_scope("densenetFe", reuse=None):
        net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
        net = densenet_block(net, nb_layers, k)
        net = densenet_block(net, nb_layers, k)
        net = densenet_block(net, nb_layers, k)
        return net