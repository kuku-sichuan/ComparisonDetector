# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf


def subtract_means(image, window, pixels_means):
    """
    :param image:(batch_size, target_side, target_side, 3)
    :param window: (y1, x1, y2, x2)
    :param pixels_means: (3)
    :return:
    """
    y1, x1, y2, x2 = np.split(window, 4)
    for i in range(np.shape(image)[0]):
        y1, x1, y2, x2 = np.split(window[i], 4)
        image[i] = image[i, y1:y2, x1:x2, :] - pixels_means
    return image


def image_resize_pad(img_tensor, gtboxes_and_label, target_side):
    """
    :param img_tensor:tensor, shape [h, w, c]
    :param gtboxes_and_label:tensor, [-1, 5]
    :param target_side : the target image shape
    :param pixel_means:
    :return:
    """
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    # resize the image size
    new_h, new_w = tf.cond(tf.greater(h, w),
                           true_fn=lambda: (target_side, target_side * w // h),
                           false_fn=lambda: (target_side * h // w,  target_side))

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)

    # resize the bbox size
    xmin, xmax = xmin * new_w // w, xmax * new_w // w
    ymin, ymax = ymin * new_h // h, ymax * new_h // h
    # the length after resizing
    new_xlen = xmax - xmin
    new_ylen = ymax - ymin
    bool_mask = tf.logical_and(tf.greater(new_xlen, 5), tf.greater(new_ylen, 5))

    # padding the image
    img_tensor = tf.squeeze(img_tensor, axis=0)
    pad_list = [compute_padding(target_side, new_h), compute_padding(target_side, new_w), [0, 0]]
    img_tensor = tf.pad(img_tensor, pad_list)

    # compute image windows(y1, x1, y2, x2)
    image_windows = tf.convert_to_tensor([pad_list[0][0], pad_list[1][0],
                                          pad_list[0][0]+new_h-1, pad_list[1][0]+new_w-1])
    # box along with the padding image
    xmin, xmax = xmin + pad_list[1][0], xmax + pad_list[1][0]
    ymin, ymax = ymin + pad_list[0][0], ymax + pad_list[0][0]
    final_bbox = tf.transpose(tf.stack([ymin, xmin, ymax, xmax, label], axis=0))
    final_valid_bbox = tf.boolean_mask(final_bbox, bool_mask)

     # ensure imgtensor rank is 3
    return img_tensor, final_valid_bbox, image_windows


def image_resize_pad_inference_data(img_tensor, target_side):

    h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.expand_dims(img_tensor, axis=0)

    new_h, new_w = tf.cond(tf.less(h, w),
                           true_fn=lambda: (target_side, target_side * w // h),
                           false_fn=lambda: (target_side * h // w, target_side))
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])
    img_tensor = tf.squeeze(img_tensor, axis=0)

    pad_list = [compute_padding(target_side, new_h), compute_padding(target_side, new_w)]
    pad_tensor = tf.convert_to_tensor(pad_list)
    img_tensor = tf.pad(img_tensor, pad_tensor)

    return img_tensor  # [1, h, w, c]


def flip_left_right(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.flip_left_right(img_tensor)

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
    new_xmin = w - xmax
    new_xmax = w - xmin
    # return img_tensor, tf.transpose(tf.stack([new_xmin, ymin, new_xmax, ymax, label], axis=0))
    return img_tensor, tf.transpose(tf.stack([ymin, new_xmin, ymax, new_xmax, label], axis=0))


def random_flip_left_right(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_left_right(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label


def compute_padding(target_side, side):
    pad_s = target_side - side
    pad_s_0 = pad_s // 2 
    pad_s_1 = pad_s - pad_s_0 
    return [pad_s_0, pad_s_1]
