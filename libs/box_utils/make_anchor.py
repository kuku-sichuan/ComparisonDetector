# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def enum_scales(base_anchor, anchor_scales, name='enum_scales'):

    """
    :param base_anchor: [y_center, x_center, h, w]
    :param anchor_scales: different scales, like [0.5, 1., 2.0]
    :return: return base anchors in different scales.
            Example:[[0, 0, 128, 128],[0, 0, 256, 256],[0, 0, 512, 512]]
    """
    with tf.variable_scope(name):
        anchor_scales = tf.reshape(anchor_scales, [-1, 1])
        return base_anchor * anchor_scales


def enum_ratios(base_anchor_size, anchor_ratios, name='enum_ratios'):

    '''
    :param base_anchor_size: base anchors size
    :param anchor_ratios: tensor,(ratios_size, ) ratio = h / w
    :return: tensor,(3, 4)[0, 0, w, h],base anchors in different scales and ratios
    '''

    with tf.variable_scope(name):
        sqrt_ratios = tf.sqrt(anchor_ratios)
        sqrt_ratios = tf.expand_dims(sqrt_ratios, axis=1)
        ws = tf.reshape(base_anchor_size / sqrt_ratios, [-1])
        hs = tf.reshape(base_anchor_size * sqrt_ratios, [-1])
        num_anchors_per_location = tf.shape(ws)[0]
        return tf.transpose(tf.stack([tf.zeros([num_anchors_per_location, ]),
                                      tf.zeros([num_anchors_per_location,]),
                                      ws, hs]))


def make_anchors(base_anchor_size, anchor_ratios, featuremaps_height,
                 featuremaps_width, feature_stride, anchor_stride, name='make_anchors'):

    """
    :param base_anchor_size: scalar, base anchor size in different scales
    :param anchor_ratios: scalar, anchor ratios
    :param featuremaps_width: scalar, width of featuremaps
    :param featuremaps_height: scalar, height of featuremaps
    :param feature_stride: scalar, a pixel in feature map == feature_stride
    :param anchor_stride: scalar, the stride of compute anchor.
    :return: tensor,(N_y*N_x, 4), [y1, x1, y2, x2] anchors of shape [w * h * len(anchor_ratios), 4]
    """
    with tf.variable_scope(name):

        base_anchors = enum_ratios(base_anchor_size, anchor_ratios)
        # ws(3,); hs(3,)
        _, _, ws, hs = tf.unstack(base_anchors, axis=1)
        # (N_x,)
        x_centers = (tf.range(0, featuremaps_width, anchor_stride, dtype=tf.float32)) * feature_stride
        # (N_y,)
        y_centers = (tf.range(0, featuremaps_height, anchor_stride, dtype=tf.float32)) * feature_stride
        # x_center(N_y, N_x)== tf.tile(x_center, N_y ,axis= 0)
        # y_center(N_y, N_x)== tf.tile(y_center, N_x ,axis= 1)
        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)
        # x_center, y_center (N_y, N_x, 3)
        # ws, hs (N_y, N_x, 3)
        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        # box_center(N_y, N_x, 3, 2)[y, x]
        box_centers = tf.stack([y_centers, x_centers], axis=2)
        box_centers = tf.reshape(box_centers, [-1, 2])

        # box_sizes(N_y, N_x, 3, 2)[h, w]
        box_sizes = tf.stack([hs, ws], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        # final_anchors(N_y*N_x, 4)[y1, x1, y2, x2]
        final_anchors = tf.concat([box_centers - 0.5*box_sizes, box_centers+0.5*box_sizes], axis=1)
        return final_anchors


def generate_pyramid_anchors(config):
    with tf.variable_scope('make_anchors'):
        anchor_list = []
        anchor_ratios = tf.constant(config.ANCHOR_RATIOS, tf.float32)
        for i in range(len(config.BACKBONE_SHAPES)):
            tmp_anchors = make_anchors(config.BASE_ANCHOR_SIZE_LIST[i], anchor_ratios,
                                       config.BACKBONE_SHAPES[i][0],
                                       config.BACKBONE_SHAPES[i][1],
                                       config.BACKBONE_STRIDES[i],
                                       config.RPN_ANCHOR_STRIDE,
                                       name='make_anchors_P{}'.format(i+2))
            tmp_anchors = tf.reshape(tmp_anchors, [-1, 4])
            anchor_list.append(tmp_anchors)

        all_level_anchors = tf.concat(anchor_list, axis=0)
        return all_level_anchors


if __name__ == '__main__':
    base_anchor = tf.constant([256], dtype=tf.float32)
    anchor_scales = tf.constant([1.0], dtype=tf.float32)
    anchor_ratios = tf.constant([0.5, 1.0, 2.0], dtype=tf.float32)
    # print(enum_scales(base_anchor, anchor_scales))
    sess = tf.Session()
    # print(sess.run(enum_ratios(enum_scales(base_anchor, anchor_scales), anchor_ratios)))
    anchors = make_anchors(256,  anchor_ratios,
                           featuremaps_height=38,
                           featuremaps_width=50,
                           feature_stride=16,
                           anchor_stride=1)

    _anchors = sess.run(anchors)
    print(_anchors)