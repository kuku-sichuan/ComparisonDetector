# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys

sys.path.append('../')
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.networks.network_factory import get_network_byname
from libs import build_rpn, build_fast_rcnn, build_fpn
from reference import load_reference_image


def model_fn(features,
             mode,
             params,
             config):
    # ***********************************************************************************************
    # *                                         share net                                           *
    # ***********************************************************************************************
    net_config = params["net_config"]
    IS_TRAINING = False

    origin_image_batch = features["image"]
    image_window = features["image_window"]
    image_batch = origin_image_batch - net_config.PIXEL_MEANS
    # there is is_training means that bn is training, so it is important!
    _, share_net = get_network_byname(inputs=image_batch,
                                      config=net_config,
                                      is_training=IS_TRAINING,
                                      reuse=tf.AUTO_REUSE)
    # ***********************************************************************************************
    # *                                            fpn                                              *
    # ***********************************************************************************************
    feature_pyramid = build_fpn.build_feature_pyramid(share_net, net_config)
    # ***********************************************************************************************
    # *                                            rpn                                              *
    # ***********************************************************************************************
    rpn = build_rpn.RPN(feature_pyramid=feature_pyramid,
                        image_window=image_window,
                        config=net_config)
    # rpn_proposals_scores==(2000,)
    rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals(IS_TRAINING)
    # ***********************************************************************************************
    # *                                         Rerference image                                           *
    # ***********************************************************************************************
    reference_image = load_reference_image()
    reference_image = tf.cast(reference_image, tf.float32)
    reference_image = reference_image - net_config.PIXEL_MEANS
    _, reference_share_net = get_network_byname(inputs=reference_image,
                                                config=net_config,
                                                is_training=False,
                                                reuse=tf.AUTO_REUSE)
    reference_feature_pyramid = build_fpn.build_feature_pyramid(reference_share_net, net_config)
    # average the features of support images
    # reference_feature_pyramid[key](C*S, H, W, 256)---->(C, 7, 7, 256)
    with tf.variable_scope('reference_feature_origision'):
        for key, value in reference_feature_pyramid.items():
            reference_feature_pyramid[key] = tf.image.resize_bilinear(reference_feature_pyramid[key],
                                                                      (net_config.ROI_SIZE, net_config.ROI_SIZE))

            reference_feature_pyramid[key] = tf.reduce_mean(tf.reshape(reference_feature_pyramid[key],
                                                            (net_config.NUM_CLASS-1, net_config.NUM_SUPPROTS,
                                                             net_config.ROI_SIZE, net_config.ROI_SIZE,
                                                             256)), axis=1)
        # average the features of fpn features
        average_fpn_feature = []
        for key, value in reference_feature_pyramid.items():
            average_fpn_feature.append(value)
        reference_fpn_features = tf.reduce_mean(tf.stack(average_fpn_feature, axis=0), axis=0)
        # compute the negative features
        with tf.variable_scope("reference_negative"):
            with slim.arg_scope([slim.conv2d],
                                padding="SAME",
                                weights_initializer=tf.glorot_uniform_initializer(),
                                weights_regularizer=slim.l2_regularizer(net_config.WEIGHT_DECAY)):
                # the shape of positive features is (1, H, W, C*channels)
                positive_features = tf.reshape(tf.transpose(reference_fpn_features, (1, 2, 0, 3)),
                                    (1, net_config.ROI_SIZE, net_config.ROI_SIZE, (net_config.NUM_CLASS-1)*256))
                # (1, H, W, channels)
                negative_feature = slim.conv2d(positive_features, num_outputs=256, kernel_size=[3,3], stride=1)
                total_refernece_feature = tf.concat([negative_feature, reference_fpn_features], axis=0)
                
    # ***********************************************************************************************
    # *                                         Fast RCNN                                           *
    # ***********************************************************************************************

    fast_rcnn = build_fast_rcnn.FastRCNN(feature_pyramid=feature_pyramid,
                                         rpn_proposals_boxes=rpn_proposals_boxes,
                                         origin_image=origin_image_batch,
                                         gtboxes_and_label=None,
                                         reference_feature=total_refernece_feature,
                                         config=net_config,
                                         is_training=IS_TRAINING,
                                         image_window=image_window)

    detections = fast_rcnn.fast_rcnn_detection()
    # ***********************************************************************************************
    # *                                          Summary                                            *
    # ***********************************************************************************************

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicts = {"image": origin_image_batch,
                    "predict_bbox": detections[:, :, :4],
                    "predict_class_id": detections[:, :, 4], "predict_scores": detections[:, :, 5],
                    "rpn_proposal_boxes": rpn_proposals_boxes,
                    "rpn_proposals_scores":rpn_proposals_scores,
                    "gt_box_labels": features["gt_box_labels"]}

        return tf.estimator.EstimatorSpec(mode, predictions=predicts)

