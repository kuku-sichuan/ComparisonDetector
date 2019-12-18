# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from libs.box_utils import encode_and_decode
from help_utils.help_utils import print_tensors


def clip_boxes_to_img_boundaries(decode_boxes, windows):
    """
    :param decode_boxes:
    :param windows: (y1, x1, y2 ,x2) the truth of image boundary
    :return: decode boxes, and already clip to boundaries
    """

    # xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(decode_boxes))
    with tf.name_scope('clip_boxes_to_img_boundaries'):

        image_y1, image_x1, image_y2, image_x2 = tf.split(tf.cast(windows, dtype=tf.float32), 4)
        ymin, xmin, ymax, xmax = tf.unstack(decode_boxes, axis=1)

        xmin = tf.maximum(xmin, tf.cast(image_x1, tf.float32))
        ymin = tf.maximum(ymin, tf.cast(image_y1, tf.float32))

        xmax = tf.minimum(xmax, tf.cast(image_x2, tf.float32))
        ymax = tf.minimum(ymax, tf.cast(image_y2, tf.float32))

        return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))


def filter_outside_boxes(boxes, windows):
    """
    compute the index boxes which is out of image boundary entirely
    :param boxes: boxes with format [ymin, xmin, ymax, xmax]
    :param windows: (y1, x1, y2 ,x2) the truth of image boundary
    :return: indices of anchors that not inside the image boundary
    """

    with tf.name_scope('filter_outside_boxes'):
        image_y1, image_x1, image_y2, image_x2 = tf.split(tf.cast(windows, dtype=tf.float32), 4)

        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        xmin_index = tf.less(xmin, image_x1)
        ymin_index = tf.less(ymin, image_y2)
        xmax_index = tf.greater(xmax, image_x1)
        ymax_index = tf.greater(ymax, image_y1)
        y_great_index = tf.greater(ymax, ymin)
        x_great_index = tf.greater(xmax, xmin)

        indices = tf.transpose(tf.stack([ymin_index, xmin_index, ymax_index, xmax_index, y_great_index, x_great_index]))
        indices = tf.cast(indices, dtype=tf.int32)
        indices = tf.reduce_sum(indices, axis=1)
        indices = tf.where(tf.greater(indices, 2))

        return tf.reshape(indices, [-1, ])


def nms_boxes(decode_boxes, scores, iou_threshold, max_output_size, name):
    '''
    NMS
    :return: valid_indices
    '''

    # xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(decode_boxes))
    # ymin, xmin, ymax, xmax = tf.unstack(decode_boxes, axis=1)
    valid_index = tf.image.non_max_suppression(
        boxes=decode_boxes,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        name=name
    )

    return valid_index


def padd_boxes_with_zeros(boxes, scores, max_num_of_boxes):

    '''
    num of boxes less than max num of boxes, so it need to pad with zeros[0, 0, 0, 0]
    :param boxes:
    :param scores: [-1]
    :param max_num_of_boxes:
    :return:
    '''

    pad_num = tf.cast(max_num_of_boxes, tf.int32) - tf.shape(boxes)[0]

    zero_boxes = tf.zeros(shape=[pad_num, 4], dtype=boxes.dtype)
    zero_scores = tf.zeros(shape=[pad_num], dtype=scores.dtype)

    final_boxes = tf.concat([boxes, zero_boxes], axis=0)

    final_scores = tf.concat([scores, zero_scores], axis=0)

    return final_boxes, final_scores


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def iou_calculate(boxes_1, boxes_2):
    '''

    :param boxes_1: (N, 4) [ymin, xmin, ymax, xmax]
    :param boxes_2: (M, 4) [ymin, xmin. ymax, xmax]
    :return:(N, M)
    '''
    with tf.name_scope('iou_caculate'):

        ymin_1, xmin_1, ymax_1, xmax_1 = tf.split(boxes_1, 4, axis=1)  # ymin_1 shape is [N, 1]..

        ymin_2, xmin_2, ymax_2, xmax_2 = tf.unstack(boxes_2, axis=1)  # ymin_2 shape is [M, ]..

        max_xmin = tf.maximum(xmin_1, xmin_2)
        min_xmax = tf.minimum(xmax_1, xmax_2)

        max_ymin = tf.maximum(ymin_1, ymin_2)
        min_ymax = tf.minimum(ymax_1, ymax_2)

        overlap_h = tf.maximum(0., min_ymax - max_ymin)  # avoid h < 0
        overlap_w = tf.maximum(0., min_xmax - max_xmin)

        overlaps = overlap_h * overlap_w

        area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
        area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

        iou = overlaps / (area_1 + area_2 - overlaps)

        return iou


def non_maximal_suppression(boxes, scores, iou_threshold, max_output_size, name='non_maximal_suppression'):
    with tf.variable_scope(name):
        nms_index = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            name=name
        )
        return nms_index


def build_rpn_target(gt_boxes, anchors, config):

    """
    assign anchors targets: object or background.
    :param anchors: (all_anchors, 4)[y1, x1, y2, x2]. use N to represent all_anchors
    :param gt_boxes: (M, 4).
    :param config: the config of making data

    :return:
    """
    with tf.variable_scope('rpn_find_positive_negative_samples'):
        gt_boxes = tf.cast(gt_boxes, tf.float32)
        ious = iou_calculate(anchors, gt_boxes)  # (N, M)

        # an anchor that has an IoU overlap higher than 0.7 with any ground-truth box
        max_iou_each_row = tf.reduce_max(ious, axis=1)
        rpn_labels = tf.ones(shape=[tf.shape(anchors)[0], ], dtype=tf.float32) * (-1)  # [N, ] # ignored is -1
        matchs = tf.cast(tf.argmax(ious, axis=1), tf.int32)
        positives1 = tf.greater_equal(max_iou_each_row, config.RPN_IOU_POSITIVE_THRESHOLD)

        # the anchor/anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box
        max_iou_each_column = tf.reduce_max(ious, 0)  # (M, )
        positives2 = tf.reduce_sum(tf.cast(tf.equal(ious, max_iou_each_column), tf.float32), axis=1)

        positives = tf.logical_or(positives1, tf.cast(positives2, tf.bool))
        rpn_labels += 2 * tf.cast(positives, tf.float32)

        anchors_matched_gtboxes = tf.gather(gt_boxes, matchs)  # [N, 4]

        # background's gtboxes tmp set the first gtbox, it dose not matter, because use object_mask will ignored it
        negatives = tf.less(max_iou_each_row, config.RPN_IOU_NEGATIVE_THRESHOLD)
        rpn_labels = rpn_labels + tf.cast(negatives, tf.float32)  # [N, ] positive is >=1.0, negative is 0, ignored is -1.0
        '''
        Need to note: when positive, labels may >= 1.0.
        Because, when all the iou< 0.7, we set anchors having max iou each column as positive.
        these anchors may have iou < 0.3.
        In the begining, labels is [-1, -1, -1...-1]
        then anchors having iou<0.3 as well as are max iou each column will be +1.0.
        when decide negatives, because of iou<0.3, they add 1.0 again.
        So, the final result will be 2.0

        So, when opsitive, labels may in [1.0, 2.0]. that is labels >=1.0
        '''
        positives = tf.cast(tf.greater_equal(rpn_labels, 1.0), tf.float32)
        ignored = tf.cast(tf.equal(rpn_labels, -1.0), tf.float32) * -1

        rpn_labels = positives + ignored

    with tf.variable_scope('rpn_minibatch'):
        # random choose the positive objects
        positive_indices = tf.reshape(tf.where(tf.equal(rpn_labels, 1.0)), [-1])  # use labels is same as object_mask
        num_of_positives = tf.minimum(tf.shape(positive_indices)[0],
                                      tf.cast(config.RPN_MINIBATCH_SIZE * config.RPN_POSITIVE_RATE,
                                              tf.int32))
        positive_indices = tf.random_shuffle(positive_indices)
        positive_indices = tf.slice(positive_indices,
                                    begin=[0],
                                    size=[num_of_positives])
        # random choose the negative objects
        negatives_indices = tf.reshape(tf.where(tf.equal(rpn_labels, 0.0)), [-1])
        num_of_negatives = tf.minimum(config.RPN_MINIBATCH_SIZE - num_of_positives,
                                      tf.shape(negatives_indices)[0])
        negatives_indices = tf.random_shuffle(negatives_indices)
        negatives_indices = tf.slice(negatives_indices, begin=[0], size=[num_of_negatives])

        minibatch_indices = tf.concat([positive_indices, negatives_indices], axis=0)

        # padding the negative objects if need
        gap = config.RPN_MINIBATCH_SIZE - tf.shape(minibatch_indices)[0]
        extract_indices = tf.random_shuffle(negatives_indices)
        extract_indices = tf.slice(extract_indices, begin=[0], size=[gap])
        minibatch_indices = tf.concat([minibatch_indices, extract_indices], axis=0)

        minibatch_indices = tf.random_shuffle(minibatch_indices)
        # (config.RPN_MINI_BATCH_SIZE, 4)
        minibatch_anchor_matched_gtboxes = tf.gather(anchors_matched_gtboxes, minibatch_indices)
        rpn_labels = tf.cast(tf.gather(rpn_labels, minibatch_indices), tf.int32)
        # encode gtboxes
        minibatch_anchors = tf.gather(anchors, minibatch_indices)
        minibatch_encode_gtboxes = encode_and_decode.encode_boxes(unencode_boxes=minibatch_anchor_matched_gtboxes,
                                                                  reference_boxes=minibatch_anchors,
                                                                  dev_factors=config.RPN_BBOX_STD_DEV)
        rpn_labels_one_hot = tf.one_hot(rpn_labels, 2, axis=-1)
        
    return minibatch_indices, minibatch_encode_gtboxes, rpn_labels_one_hot


def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)