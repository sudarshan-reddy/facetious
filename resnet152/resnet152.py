# -*- coding: utf-8 -*-

import numpy as np
import copy

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K

import sys
sys.setrecursionlimit(3000)

class Scale(Layer):
    '''Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet152_model(img_rows, img_cols, color_type=1, weights_path=None, load_top=False, new_top=False):
    '''Instantiate the ResNet152 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')
            
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x_feature_extractor_end = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_classifier = AveragePooling2D((7, 7), name='avg_pool')(x_feature_extractor_end)
    x_classifier = Flatten()(x_classifier)
    x_classifier = Dense(1000, activation='softmax', name='fc1000')(x_classifier)

    model = Model(img_input, x_classifier)
    
    # load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    if not load_top:
        # Truncate and replace softmax layer for transfer learning
        # Cannot use model.layers.pop() since model is not of Sequential() type
        # The method below works since pre-trained weights are stored in layers but not in the model
        if new_top:
            x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x_feature_extractor_end)
            x_newfc = Flatten()(x_newfc)
            x_newfc = Dense(1, activation='sigmoid', name='fc8')(x_newfc)
            model = Model(img_input, x_newfc)
        else:
            model = Model(img_input, x_feature_extractor_end)

    return model

if __name__ == '__main__':
    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    batch_size = 8
    nb_epoch = 10


    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'resnet152_weights_th.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'resnet152_weights_tf.h5'

    # Test pretrained model
    model = resnet152_model(img_rows, img_cols, channel, weights_path, load_top=False, new_top=True)

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    model.predict("test")

    #from keras.utils import plot_model
    #plot_model(model, to_file='plotted_model.png', show_shapes=True)

    #print "[*] We can realistically cut at these layers:"
    #for layer in model.layers:
    #    if "Add" in type(layer).__name__:
    #        print layer.name, layer, layer.get_config()

"""
res2a <keras.layers.merge.Add object at 0x7fc294b993d0> {'trainable': True, 'name': 'res2a'}
res2b <keras.layers.merge.Add object at 0x7fc294a22710> {'trainable': True, 'name': 'res2b'}
res2c <keras.layers.merge.Add object at 0x7fc2948b75d0> {'trainable': True, 'name': 'res2c'}
res3a <keras.layers.merge.Add object at 0x7fc294655bd0> {'trainable': True, 'name': 'res3a'}
res3b1 <keras.layers.merge.Add object at 0x7fc2944e6290> {'trainable': True, 'name': 'res3b1'}
...
res3b7 <keras.layers.merge.Add object at 0x7fc293b62250> {'trainable': True, 'name': 'res3b7'}
res4a <keras.layers.merge.Add object at 0x7fc293967c90> {'trainable': True, 'name': 'res4a'}
res4b1 <keras.layers.merge.Add object at 0x7fc293786610> {'trainable': True, 'name': 'res4b1'}
...
res4b34 <keras.layers.merge.Add object at 0x7fc2903bafd0> {'trainable': True, 'name': 'res4b34'}
res4b35 <keras.layers.merge.Add object at 0x7fc2901f7e50> {'trainable': True, 'name': 'res4b35'}
res5a <keras.layers.merge.Add object at 0x7fc28ffe4d90> {'trainable': True, 'name': 'res5a'}
res5b <keras.layers.merge.Add object at 0x7fc28fe7b1d0> {'trainable': True, 'name': 'res5b'}
res5c <keras.layers.merge.Add object at 0x7fc28fc90690> {'trainable': True, 'name': 'res5c'}
"""

"""
____________________________________________________________________________________________________
scale5c_branch2b (Scale)         (None, 7, 7, 512)     1024
____________________________________________________________________________________________________
res5c_branch2b_relu (Activation) (None, 7, 7, 512)     0
____________________________________________________________________________________________________
res5c_branch2c (Conv2D)          (None, 7, 7, 2048)    1048576
____________________________________________________________________________________________________
bn5c_branch2c (BatchNormalizatio (None, 7, 7, 2048)    8192
____________________________________________________________________________________________________
scale5c_branch2c (Scale)         (None, 7, 7, 2048)    4096
____________________________________________________________________________________________________
res5c (Add)                      (None, 7, 7, 2048)    0
____________________________________________________________________________________________________
res5c_relu (Activation)          (None, 7, 7, 2048)    0
____________________________________________________________________________________________________
avg_pool (AveragePooling2D)      (None, 1, 1, 2048)    0
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2048)          0
____________________________________________________________________________________________________
fc1000 (Dense)                   (None, 1000)          2049000
<< original top
====================================================================================================
Total params: 60,495,656
Trainable params: 60,344,232
Non-trainable params: 151,424


res5c (Add)                      (None, 7, 7, 2048)    0
____________________________________________________________________________________________________
res5c_relu (Activation)          (None, 7, 7, 2048)    0
<< without top
====================================================================================================
Total params: 58,446,656
Trainable params: 58,295,232
Non-trainable params: 151,424


res5c (Add)                      (None, 7, 7, 2048)    0
____________________________________________________________________________________________________
res5c_relu (Activation)          (None, 7, 7, 2048)    0
____________________________________________________________________________________________________
avg_pool (AveragePooling2D)      (None, 1, 1, 2048)    0
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 2048)          0
____________________________________________________________________________________________________
fc8 (Dense)                      (None, 1)             2049
<< cutom top (for linear regression problem - we have sigmoid)
====================================================================================================
Total params: 58,448,705
Trainable params: 58,297,281
Non-trainable params: 151,424

"""
