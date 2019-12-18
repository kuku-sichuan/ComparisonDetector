# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys

sys.path.append('../')
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline

from libs.networks.network_factory import get_network_byname
from libs import build_rpn, build_fast_rcnn, build_fpn
from libs.box_utils.show_box_in_tensor import draw_boxes_with_scores, draw_boxes_with_categories_and_scores
from data.io.read_tfrecord import train_input_fn
from help_utils.help_utils import print_tensors
from configs.config import Config
from reference import load_reference_image
DEBUG = True


def model_fn(features,
             labels,
             mode,
             params,
             config):

    # ***********************************************************************************************
    # *                                         share net                                           *
    # ***********************************************************************************************
    net_config = params["net_config"]
    if mode == tf.estimator.ModeKeys.TRAIN:
        IS_TRAINING = True
    else:
        IS_TRAINING = False

    origin_image_batch = features["image"]
    image_window = features["image_window"]
    image_batch = origin_image_batch - net_config.PIXEL_MEANS
    # there is is_training means that bn is training, so it is important!
    _, share_net = get_network_byname(inputs=image_batch,
                                      config=net_config,
                                      is_training=False,
                                      reuse=tf.AUTO_REUSE)
    # ***********************************************************************************************
    # *                                            fpn                                              *
    # ***********************************************************************************************
    feature_pyramid = build_fpn.build_feature_pyramid(share_net, net_config)
    # ***********************************************************************************************
    # *                                            rpn                                              *
    # ***********************************************************************************************
    gtboxes_and_label_batch = labels.get("gt_box_labels")
    rpn = build_rpn.RPN(feature_pyramid=feature_pyramid,
                        image_window=image_window,
                        config=net_config)

    # rpn_proposals_scores==(2000,)
    rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals(IS_TRAINING)
    rpn_location_loss, rpn_classification_loss = rpn.rpn_losses(labels["minibatch_indices"],
                                                                labels["minibatch_encode_gtboxes"],
                                                                labels["minibatch_objects_one_hot"])
        
    rpn_total_loss = rpn_classification_loss + rpn_location_loss

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
                                         gtboxes_and_label=gtboxes_and_label_batch,
                                         reference_feature=total_refernece_feature,
                                         config=net_config,
                                         is_training=False,
                                         image_window=image_window)

    detections = fast_rcnn.fast_rcnn_detection()
    if DEBUG:
        rpn_proposals_vision = draw_boxes_with_scores(origin_image_batch[0, :, :, :],
                                                      rpn_proposals_boxes[0, :50, :],
                                                      rpn_proposals_scores[0, :50])
        fast_rcnn_vision = draw_boxes_with_categories_and_scores(origin_image_batch[0, :, :, :],
                                                                 detections[0, :, :4],
                                                                 detections[0, :, 4],
                                                                 detections[0, :, 5])
        tf.summary.image("rpn_proposals_vision", rpn_proposals_vision) 
        tf.summary.image("fast_rcnn_vision", fast_rcnn_vision)

    fast_rcnn_location_loss, fast_rcnn_classification_loss = fast_rcnn.fast_rcnn_loss()
    fast_rcnn_total_loss = 5.0*fast_rcnn_classification_loss + fast_rcnn_location_loss

    # train
    with tf.variable_scope("regularization_losses"):
        regularization_list = [tf.nn.l2_loss(w.read_value()) *
                               net_config.WEIGHT_DECAY / tf.cast(tf.size(w.read_value()),
                               tf.float32) for w in tf.trainable_variables() if 'gamma' not
                               in w.name and 'beta' not in w.name]
        regularization_losses = tf.add_n(regularization_list)

    total_loss = regularization_losses + fast_rcnn_total_loss + rpn_total_loss
    global_step = slim.get_or_create_global_step()
    tf.train.init_from_checkpoint(net_config.CHECKPOINT_DIR, {net_config.NET_NAME + "/": net_config.NET_NAME + "/"})
    with tf.variable_scope("optimizer"):
        lr = tf.train.piecewise_constant(global_step,
                                         boundaries=[np.int64(net_config.BOUNDARY[0]), np.int64(net_config.BOUNDARY[1])],
                                         values=[net_config.LEARNING_RATE, net_config.LEARNING_RATE / 10,
                                                 net_config.LEARNING_RATE / 100])
        optimizer = tf.train.MomentumOptimizer(lr, momentum=net_config.MOMENTUM)
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            grads = optimizer.compute_gradients(total_loss)
            for i, (g, v) in enumerate(grads):
                if g is not None:
                    grads[i] = (tf.clip_by_norm(g, 5.0), v)  # clip gradients
            train_op = optimizer.apply_gradients(grads, global_step)

    # ***********************************************************************************************
    # *                                          Summary                                            *
    # ***********************************************************************************************
    # rpn loss and image
    tf.summary.scalar('rpn/rpn_location_loss', rpn_location_loss)
    tf.summary.scalar('rpn/rpn_classification_loss', rpn_classification_loss)
    tf.summary.scalar('rpn/rpn_total_loss', rpn_total_loss)

    tf.summary.scalar('fast_rcnn/fast_rcnn_location_loss', fast_rcnn_location_loss)
    tf.summary.scalar('fast_rcnn/fast_rcnn_classification_loss', fast_rcnn_classification_loss)
    tf.summary.scalar('fast_rcnn/fast_rcnn_total_loss', fast_rcnn_total_loss)
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('total_loss', total_loss)

    summary_hook = tf.train.SummarySaverHook(save_steps=net_config.SAVE_EVERY_N_STEP,
                                             output_dir=net_config.MODLE_DIR,
                                             summary_op=tf.summary.merge_all())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op, training_hooks=[summary_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        predicts = {"predict_bbox": detections[:, :, :4],
                    "predict_class_id": detections[:, :, 5], "predict_scores": detections[:, :, 4]}
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, predictions=predicts)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicts = {"predict_bbox": detections[:, :, :4],
                    "predict_class_id": detections[:, :, 5], "predict_scores": detections[:, :, 4]}

        return tf.estimator.EstimatorSpec(mode, predictions=predicts)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    net_config = Config()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    estimator_config = tf.estimator.RunConfig(model_dir=net_config.MODLE_DIR,
                                              save_summary_steps=net_config.SAVE_EVERY_N_STEP,
                                              save_checkpoints_steps=net_config.SAVE_EVERY_N_STEP,
                                              session_config=session_config)

    my_estimator = tf.estimator.Estimator(tf.contrib.estimator.replicate_model_fn(model_fn,
                                                                             devices=net_config.GPU_GROUPS),
                                      params={"net_config": net_config}, 
                                      config=estimator_config)
    my_estimator.train(input_fn=lambda: train_input_fn(net_config.DATA_DIR, net_config.BATCH_SIZE, net_config.EPOCH))




