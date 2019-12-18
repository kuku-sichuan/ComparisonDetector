# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def decode_boxes(encode_boxes, reference_boxes, dev_factors=None, name='decode'):
    '''

    :param encode_boxes:[N, 4]
    :param reference_boxes: [N, 4] .
    :param dev_factors: use for scale
    in the first stage, reference_boxes  are anchors
    in the second stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 4]
    '''

    with tf.name_scope(name):
        t_ycenter, t_xcenter, t_h, t_w = tf.unstack(encode_boxes, axis=1)
        if dev_factors:
            t_xcenter *= dev_factors[0]
            t_ycenter *= dev_factors[1]
            t_w *= dev_factors[2]
            t_h *= dev_factors[3]

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)

        reference_w = reference_xmax - reference_xmin + 1
        reference_h = reference_ymax - reference_ymin + 1

        reference_xcenter = reference_xmin + 0.5 * reference_w
        reference_ycenter = reference_ymin + 0.5 * reference_h

        predict_xcenter = t_xcenter * reference_w + reference_xcenter
        predict_ycenter = t_ycenter * reference_h + reference_ycenter
        predict_w = tf.exp(t_w) * reference_w
        predict_h = tf.exp(t_h) * reference_h

        predict_xmin = predict_xcenter - predict_w / 2.
        predict_xmax = predict_xcenter + predict_w / 2.
        predict_ymin = predict_ycenter - predict_h / 2.
        predict_ymax = predict_ycenter + predict_h / 2.

        return tf.transpose(tf.stack([predict_ymin, predict_xmin,
                                      predict_ymax, predict_xmax]))


def encode_boxes(unencode_boxes, reference_boxes, dev_factors=None, name='encode'):
    '''

    :param unencode_boxes: [ N, 4](gt_box)
    :param reference_boxes: [N, 4](anchors)
    :return: encode_boxes [-1, 4]
    '''

    with tf.variable_scope(name):
        ymin, xmin, ymax, xmax = tf.unstack(unencode_boxes, axis=1)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        x_center = xmin + 0.5 * w
        y_center = ymin + 0.5 * h

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)

        reference_w = reference_xmax - reference_xmin + 1
        reference_h = reference_ymax - reference_ymin + 1
        reference_xcenter = reference_xmin + 0.5 * reference_w
        reference_ycenter = reference_ymin + 0.5 * reference_h

        t_xcenter = (x_center - reference_xcenter) / reference_w
        t_ycenter = (y_center - reference_ycenter) / reference_h
        t_w = tf.log(w / reference_w)
        t_h = tf.log(h / reference_h)

        if dev_factors:
            t_xcenter /= dev_factors[0]
            t_ycenter /= dev_factors[1]
            t_w /= dev_factors[2]
            t_h /= dev_factors[3]

        return tf.transpose(tf.stack([t_ycenter, t_xcenter, t_h, t_w]))
