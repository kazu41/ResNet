from __future__ import print_function
import numpy as np
from functools import reduce

import tensorflow as tf

from keras import backend as K
from keras.layers import (Input, Activation, Add, Lambda,
                          BatchNormalization, Conv2D, Dense, Flatten, Reshape,
                          MaxPooling2D, GlobalAveragePooling2D)
from keras.models import Model
from keras.regularizers import l2

K.set_image_data_format("channels_last")
K.set_image_dim_ordering("tf")
ROW_AXIS, COL_AXIS, CHANNEL_AXIS = 1, 2, 3

def compose(*funcs):
    '''
    compose multiple layers
    '''
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def ResNetConv2D(*args, **kwargs):
    '''
    helper func for Conv2D
    '''
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(1.e-4)
    }
    conv_kwargs.update(kwargs)

    return Conv2D(*args, **conv_kwargs)

def bn_relu_conv(*args, **kwargs):
    '''
    helper func to provide
    batch mormalization -> ReLU -> conv
    '''
    return compose(
        BatchNormalization(),
        Activation('relu'),
        ResNetConv2D(*args, **kwargs))

def shortcut(x, y, mode="zero-padding"):
    '''
    shortcut connection

    Arguments
    ==============
    mode: zero-padding or projection
    '''
    assert mode in ("zero-padding", "projection")

    x_shape = K.int_shape(x)
    y_shape = K.int_shape(y)

    if x_shape == y_shape:
        shortcut = x

    else:
        stride_w = int(round(x_shape[ROW_AXIS] / y_shape[ROW_AXIS]))
        stride_h = int(round(x_shape[COL_AXIS] / y_shape[COL_AXIS]))

        if mode=="projection":
            shortcut = Conv2D(filters=y_shape[CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_w, stride_h),
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(1.e-4))(x)

        elif mode=="zero-padding":
            x = Lambda(lambda x: x[:, ::stride_w, ::stride_h])(x) # downsampling
            shortcut = Lambda(lambda x: tf.concat((x, tf.zeros_like(y[:,:,:,x_shape[CHANNEL_AXIS]:])), axis=CHANNEL_AXIS))(x)

    return Add()([shortcut, y])

def basic_block(filters, first_strides, is_first_block_of_first_layer, mode="zero-padding"):
    '''
    provide bulding block func

    Arguments
    ===============================
    filters: number of filters
    first_strides: strides of the first convolution
    is_first_block_of_first_layer: Boolean indicating the first res-block after max pooling
    mode: shortcut mode, zero-padding or projection

    '''
    def f(x):
        if is_first_block_of_first_layer:
            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                 strides=first_strides)(x)

        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        return shortcut(x, conv2, mode=mode)

    return f

def bottleneck_block(filters, first_strides, is_first_block_of_first_layer, mode="zero-padding"):
    '''
    provide bottleneck bulding block func

    Arguments
    ===============================
    filters: number of filters
    first_strides: strides of the first convolution
    is_first_block_of_first_layer: Boolean indicating the first res-block after max pooling
    mode: shortcut mode, zero-padding or projection

    '''
    def f(x):
        if is_first_block_of_first_layer:
            conv1 = ResNetConv2D(filters=filters//4, kernel_size=(1, 1))(x)
        else:
            conv1 = bn_relu_conv(filters=filters//4, kernel_size=(1, 1),
                                 strides=first_strides)(x)

        conv2 = bn_relu_conv(filters=filters//4, kernel_size=(3, 3))(conv1)
        conv3 = bn_relu_conv(filters=filters, kernel_size=(1, 1))(conv2)

        return shortcut(x, conv3, mode=mode)

    return f

def residual_blocks(block_function, filters, repetitions, is_first_layer, mode="zero-padding"):
    '''
    return multiple residual blocks

    Arguments
    ===================================
    block_function: a function of residual block
    filters: number of filters
    repetitions: of residual blocks
    is_first_layer: Boolean indicating if the first layer after max pooling
    '''
    def f(x):
        for i in range(repetitions):
            # conv3_x, conv4_x, conv5_x の最初の畳み込みは、
            # プーリング目的の畳み込みなので、strides を (2, 2) にする。
            # ただし、conv2_x の最初の畳み込みは直前の max pooling 層でプーリングしているので
            # strides を (1, 1) にする。
            first_strides = (2, 2) if i == 0 and not is_first_layer else (1, 1)

            x = block_function(filters=filters, first_strides=first_strides,
                               is_first_block_of_first_layer=(i == 0 and is_first_layer),
                               mode=mode)(x)
        return x

    return f

class ResnetBuilder():
    @staticmethod
    def build(input_shape, num_outputs, block_type, repetitions, wide=False, mode="projection"):
        '''
        ResNet モデルを作成する Factory クラス

        Arguments
        ==============================
        input_shape
        num_outputs
        block_type : residual block の種類 ('basic' or 'bottleneck')
        repetitions: 同じ residual block を何個反復させるか
        wide:
        mode:
        '''
        # block_type に応じて、residual block を生成する関数を選択する。
        if block_type == 'basic':
            block_fn = basic_block
        elif block_type == 'bottleneck':
            block_fn = bottleneck_block

        # モデルを作成する。
        ##############################################
        input = Input(shape=input_shape)

        # conv1 (batch normalization -> ReLU -> conv)
        conv1 = compose(ResNetConv2D(filters=16, kernel_size=(7, 7), strides=(2, 2)),
                        BatchNormalization(),
                        Activation('relu'))(input)

        # pool
        pool1 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

        # conv2_x, conv3_x, conv4_x, conv5_x
        block = pool1
        filters = 32 if wide==False else 64
        for i, r in enumerate(repetitions):
            block = residual_blocks(block_fn, filters=filters,
                                    repetitions=r,is_first_layer=(i == 0),
                                    mode=mode)(block)
            filters *= 2

        # batch normalization -> ReLU
        block = compose(BatchNormalization(),
                        Activation('relu'))(block)

        # global average pooling
        pool2 = GlobalAveragePooling2D()(block)

        # dense
        fc1 = Dense(units=num_outputs,
                    kernel_initializer='he_normal',
                    activation='softmax')(pool2)

        # Instantiate model.
        model = Model(inputs=input, outputs=fc1)
        depth = int(np.sum(repetitions)*2+2) if block_type=="basic" else int(np.sum(repetitions)*3+2)
        netname = "ResNet" if not wide else "WResNet"
        model.name = "{}{}_xshp-{}_ncls-{}".format(netname, depth, "{}x{}x{}".format(*input_shape), num_outputs)
        model.num_params = model.count_params()
        model.num_trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])

        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'basic', [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'basic', [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'bottleneck', [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'bottleneck', [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'bottleneck', [3, 8, 36, 3])

class PyramidNetBuilder():
    @staticmethod
    def build(input_shape, num_outputs, block_type, repetitions=[3,3,3], alpha=240, mode="zero-padding"):
        '''
        PyramidNet モデルを作成する Factory クラス

        Reference
        D. Han, J. Kim, and J. Kim, “Deep pyramidal residual net- works,” CoRR, 2016.
        https://arxiv.org/abs/1610.02915

        Arguments
        ==============================
        input_shape
        num_outputs
        block_type : residual block の種類 ('basic' or 'bottleneck')
        repetitions: 同じ residual block を何個反復させるか
        alpha:
        mode:
        '''
        num_blocks = np.sum(repetitions)
        filters_list = np.floor(np.linspace(16,16+alpha,num_blocks+1)).astype(np.int)

        # block_type に応じて、residual block を生成する関数を選択する。
        if block_type == 'basic':
            block_fn = basic_block
        elif block_type == 'bottleneck':
            block_fn = bottleneck_block

        # モデルを作成する。
        ##############################################
        input = Input(shape=input_shape)

        # conv1 (batch normalization -> ReLU -> conv)
        conv1 = compose(ResNetConv2D(filters=filters_list[0],
                        kernel_size=(3, 3), strides=(1, 1)),
                        BatchNormalization(),
                        Activation('relu'))(input)

        # conv2_x, conv3_x, conv4_x, conv5_x
        block = conv1

        k = 1
        for b in np.arange(len(repetitions)):
            for i in np.arange(repetitions[b]):
                first_strides = (2, 2) if i == 0 and not b==0 else (1, 1)
                block = block_fn(filters=filters_list[k], first_strides=first_strides,
                                is_first_block_of_first_layer=(i == 0 and b==0),
                                mode=mode)(block)
                k += 1

        # batch normalization -> ReLU
        block = compose(BatchNormalization(),
                        Activation('relu'))(block)

        # global average pooling
        pool2 = GlobalAveragePooling2D()(block)

        # dense
        fc1 = Dense(units=num_outputs,
                    kernel_initializer='he_normal',
                    activation='softmax')(pool2)

        # Instantiate model.
        model = Model(inputs=input, outputs=fc1)
        depth = int(np.sum(repetitions)*2+2) if block_type=="basic" else int(np.sum(repetitions)*3+2)
        netname = "PyramidNet"
        model.name = "{}{}_alpha-{}_xshp-{}_ncls-{}".format(netname, depth, alpha, "{}x{}x{}".format(*input_shape), num_outputs)
        model.num_params = model.count_params()
        model.num_trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])

        return model
