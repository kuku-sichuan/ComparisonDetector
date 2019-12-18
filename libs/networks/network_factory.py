from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
sys.path.insert(0, '../../')

from libs.networks.nets import resnet_v1
from libs.networks.nets import inception_resnet_v2
from libs.networks.nets import resnet_v2
from libs.networks.nets import means_resnet_v2
from libs.networks.nets import vgg


def get_network_byname(inputs,
                       config,
                       is_training,
                       reuse):
    if config.NET_NAME == 'resnet_v1_50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=config.WEIGHT_DECAY)):
            logits, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                        store_non_strided_activations=True,
                                                        is_training=is_training,
                                                        global_pool=False,
                                                        reuse=reuse)
        return logits, end_points

    if config.NET_NAME == 'resnet_v1_101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=config.WEIGHT_DECAY)):
            logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                         is_training=is_training,
                                                         store_non_strided_activations=True,
                                                         global_pool=False,
                                                         reuse=reuse)
        return logits, end_points
    if config.NET_NAME == 'resnet_v2_50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=config.WEIGHT_DECAY)):
            logits, end_points = resnet_v2.resnet_v2_50(inputs=inputs,
                                                        is_training=is_training,
                                                        global_pool=False,
                                                        reuse=reuse)
        return logits, end_points
    if config.NET_NAME == 'resnet_model':
        features_map = means_resnet_v2.resnet_v2(inputs=inputs,
                                                training=is_training,
                                                reuse=reuse)
        return None, features_map

    if config.NET_NAME == 'inception_resnet':
        arg_sc = inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=config.WEIGHT_DECAY)
        with slim.arg_scope(arg_sc):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs=inputs,
                                                                         is_training=is_training,
                                                                         reuse=reuse)
        return logits, end_points

    if config.NET_NAME == 'vgg16':
        arg_sc = vgg.vgg_arg_scope(weight_decay=config.WEIGHT_DECAY)
        with slim.arg_scope(arg_sc):
            logits, end_points = vgg.vgg_16(inputs=inputs,
                                            is_training=is_training,
                                            spatial_squeeze=False)
        return logits, end_points
