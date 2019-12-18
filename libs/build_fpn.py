# # -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from help_utils.help_utils import print_tensors

def build_feature_pyramid(share_net, config, reuse=tf.AUTO_REUSE):
    """
    :param share_net: the dict of network's output at every point.
    :param config: the config of network
    :param reuse:
    :return: the dict of multi-pyramid feature map {P2:,----,P6:}
    """

    feature_maps_dict ={}
    with tf.variable_scope('get_feature_maps', reuse=reuse):
        if config.NET_NAME == 'resnet_v2_50':
            feature_maps_dict = {
                'C2': share_net['resnet_v2_50/block1/unit_2/bottleneck_v2'],  # [256, 256]
                'C3': share_net['resnet_v2_50/block2/unit_3/bottleneck_v2'],  # [128, 128]
                'C4': share_net['resnet_v2_50/block3/unit_5/bottleneck_v2'],  # [64, 64]
                'C5': share_net['resnet_v2_50/block4/unit_3/bottleneck_v2']  # [32, 32]
            }
        elif config.NET_NAME == 'resnet_v2_101':
            feature_maps_dict = {
                'C2': share_net['resnet_v2_101/block1/unit_2/bottleneck_v2'],  # [56, 56]
                'C3': share_net['resnet_v2_101/block2/unit_3/bottleneck_v2'],  # [28, 28]
                'C4': share_net['resnet_v2_101/block3/unit_22/bottleneck_v2'],  # [14, 14]
                'C5': share_net['resnet_v2_101/block4/unit_3/bottleneck_v2']  # [7, 7]
            }
        elif config.NET_NAME == 'resnet_v1_50':
            feature_maps_dict = {
                'C2': share_net['resnet_v1_50/block1/unit_3/bottleneck_v1'],  # [256, 256]
                'C3': share_net['resnet_v1_50/block2/unit_4/bottleneck_v1'],  # [128, 128]
                'C4': share_net['resnet_v1_50/block3/unit_6/bottleneck_v1'],  # [64, 64]
                'C5': share_net['resnet_v1_50/block4/unit_3/bottleneck_v1']  # [32, 32]
            }
        elif config.NET_NAME == 'resnet_v1_101':
            feature_maps_dict = {
                'C2': share_net['resnet_v1_101/block1/unit_2/bottleneck_v1'],  # [56, 56]
                'C3': share_net['resnet_v1_101/block2/unit_3/bottleneck_v1'],  # [28, 28]
                'C4': share_net['resnet_v1_101/block3/unit_22/bottleneck_v1'],  # [14, 14]
                'C5': share_net['resnet_v1_101/block4/unit_3/bottleneck_v1']  # [7, 7]
            }
        elif config.NET_NAME == 'resnet_model':
            feature_maps_dict =  share_net
        else:
            raise Exception('get no feature maps')
    feature_pyramid = {}
    with tf.variable_scope('build_feature_pyramid', reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=None,
                            normalizer_params=None,
                            weights_initializer=tf.glorot_uniform_initializer(),
                            weights_regularizer=slim.l2_regularizer(config.WEIGHT_DECAY)):
            feature_pyramid['P5'] = slim.conv2d(feature_maps_dict['C5'],
                                                num_outputs=256,
                                                kernel_size=[1, 1],
                                                stride=1,
                                                scope='C5toP5')

            # P6 is down sample of P5

            for layer in range(4, 1, -1):
                p, c = feature_pyramid['P' + str(layer + 1)], feature_maps_dict['C' + str(layer)]
                up_sample_shape = tf.shape(c)
                up_sample = tf.image.resize_images(p,
                                                   [up_sample_shape[1], up_sample_shape[2]])

                c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                scope='C%dtoP%d' %(layer, layer))
                p = up_sample + c
                feature_pyramid['P' + str(layer)] = p
            for layer in range(5, 1, -1):
                p = feature_pyramid['P' + str(layer)]
                p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1,
                                padding='SAME', scope='build_P%d' % layer)
                feature_pyramid['P' + str(layer)] = p
            feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                                                    kernel_size=[2, 2], stride=2, scope='build_P6')

    return feature_pyramid


