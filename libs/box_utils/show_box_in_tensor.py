# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import colorsys
import cv2
from libs.label_dict import LABEl_NAME_MAP


def class_colors(class_ids, num_classes, bright=True):
    """
    based on the class id to choose a centrial color to show them
    """
    brightness = 1.0 if bright else 0.7
    colors = np.zeros_like(class_ids)
    hsv = [(i / np.float(num_classes), 1, brightness) for i in range(num_classes)]
    color_map = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    color_map = np.ceil(np.array(color_map) * 255)
    colors = color_map[class_ids - 1]

    return colors


def draw_box_in_img_batch(img_batch, boxes):
    """
    img_batch: (N, W, H, 3)
    boxes: (N, M, 4)
    """
    boxes = tf.cast(boxes, tf.float32)
    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=2)

    img_h, img_w = tf.shape(img_batch)[1], tf.shape(img_batch)[2]
    abs_xmin = xmin / tf.cast(img_w, tf.float32)
    abs_ymin = ymin / tf.cast(img_h, tf.float32)
    abs_xmax = xmax / tf.cast(img_w, tf.float32)
    abs_ymax = ymax / tf.cast(img_h, tf.float32)

    return tf.image.draw_bounding_boxes(img_batch,
                                        boxes=tf.transpose(tf.stack([abs_ymin, abs_xmin,
                                                                                    abs_ymax, abs_xmax]),[1, 2, 0]))


def draw_box_with_color(img_batch, boxes, text):

    def draw_box_cv(img, boxes, text):
        boxes = boxes.astype(np.int64)
        img = np.array(img * 255 / np.max(img), np.uint8)
        for box in boxes:
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

        text = str(text)
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        # img = np.transpose(img, [2, 1, 0])
        img = img[:, :, -1::-1]
        return img

    img_tensor = img_batch
    # color = tf.constant([0, 0, 255])
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text],
                                       Tout=[tf.uint8])

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    img_tensor_with_boxes = tf.expand_dims(img_tensor_with_boxes, 0)

    return img_tensor_with_boxes


def draw_boxes_with_scores(img_batch, boxes, scores):

    def draw_box_cv(img, boxes, scores):
        boxes = boxes.astype(np.int64)
        img = np.array(img, np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
            score = scores[i]

            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmin+20, ymin+10),
                          color=color,
                          thickness=-1)

            cv2.putText(img,
                        text=str(np.round(score, 2)),
                        org=(xmin, ymin+10),
                        fontFace=1,
                        fontScale=1,
                        thickness=2,
                        color=(color[1], color[2], color[0]))
            num_of_object += 1
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        return img
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_batch, boxes, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    img_tensor_with_boxes = tf.expand_dims(img_tensor_with_boxes, 0)
    return img_tensor_with_boxes


def draw_boxes_with_categories(img_batch, boxes, labels):

    def draw_box_cv(img, boxes, labels):
        boxes = boxes.astype(np.int64)
        img = np.array(img, np.uint8)

        num_of_object = 0
        color = class_colors(labels, 12)
        for i, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color[i],
                          thickness=2)

            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmin + 20, ymin + 10),
                          color=color[i],
                          thickness=-1)

            category = LABEl_NAME_MAP[labels[i]]
            cv2.putText(img,
                        text=category,
                        org=(xmin, ymin + 10),
                        fontFace=1,
                        fontScale=1,
                        thickness=2,
                        color=(255, 255, 255))
            num_of_object += 1
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        return img

    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_batch, boxes, labels],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    img_tensor_with_boxes = tf.expand_dims(img_tensor_with_boxes, 0)
    return img_tensor_with_boxes


def draw_boxes_with_categories_and_scores(img_batch, boxes, labels, scores):

    def draw_box_cv(img, boxes, labels, scores):
        boxes = boxes.astype(np.int64)
        labels = labels.astype(np.int32)
        img = np.array(img, np.uint8)
        color = class_colors(labels, 12)

        num_of_object = 0
        for i, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            label = labels[i]
            score = scores[i]
            if label != 0:
                num_of_object += 1
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmax, ymax),
                              color=color[i],
                              thickness=2)
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmin+120, ymin+15),
                              color=color[i],
                              thickness=-1)
                category = LABEl_NAME_MAP[label]
                cv2.putText(img,
                            text=category+": "+str(score),
                            org=(xmin, ymin+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(255, 255, 255))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        return img

    img_tensor = img_batch
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    img_tensor_with_boxes = tf.expand_dims(img_tensor_with_boxes, 0)
    return img_tensor_with_boxes


if __name__ == "__main__":

    img = cv2.imread('1.jpg')
    img = tf.constant(np.expand_dims(img, 0), tf.float32)
    boxes = tf.constant([[30, 30, 230, 230]])
    labels = tf.constant([1])
    scores = tf.constant([0.6])
    img_ten = draw_boxes_with_categories(img, boxes, labels, scores)

    with tf.Session() as sess:
        img_np = sess.run(img_ten)
        img_np = np.squeeze(img_np, 0)
        cv2.imshow('test', img_np)
        cv2.waitKey(0)
