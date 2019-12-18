import numpy as np
import tensorflow as tf
import math


class Config(object):

    ##############################
    # Data And Dataset
    ##############################
    CHECKPOINT_DIR= "/root/userfolder/kuku/20180601_resnet_v2_imagenet_checkpoint"
    NUM_CLASS = 11 + 1
    NUM_ITEM_DATASET = 5714
    DATASET_NAME = 'tct'
    DATA_DIR = "./tfdata"
    MODLE_DIR = "./logs"
    # resize and padding the image shape to (1024, 1024)
    TARGET_SIDE = 1024
    FAST_RCNN_MAX_INSTANCES = 100
    PIXEL_MEANS = np.array([115.2, 118.8, 123.0])
    NUM_SUPPROTS = 3

    ###################################
    # Network config
    ###################################
    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1
    NET_NAME = 'resnet_model'
    VERSION = 'v1_tct'
    BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
    LEVEL = ['P2', 'P3', 'P4', 'P5', "P6"]
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]


    ###################################
    # Training Config
    ###################################
    EPOCH_BOUNDARY = [35, 50]
    EPOCH = 60
    WEIGHT_DECAY = 0.0001
    EPSILON = 1e-5
    MOMENTUM = 0.9
    GPU_GROUPS = ["/gpu:0", "/gpu:1"]
    LEARNING_RATE = 0.001
    PER_GPU_IMAGE = 1
    CLIP_GRADIENT_NORM = 5.0

    ###################################
    # RPN
    ###################################
    ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_NMS_IOU_THRESHOLD = 0.7
    RPN_IOU_POSITIVE_THRESHOLD = 0.7
    RPN_IOU_NEGATIVE_THRESHOLD = 0.3
    RPN_MINIBATCH_SIZE = 256
    RPN_POSITIVE_RATE = 0.5
    RPN_TOP_K_NMS = 6000
    MAX_PROPOSAL_NUM_TRAINING = 2000
    MAX_PROPOSAL_NUM_INFERENCE = 1000
    RPN_BBOX_STD_DEV = [0.1, 0.1, 0.25, 0.27]
    BBOX_STD_DEV = [0.13, 0.13, 0.27, 0.26]

    ###################################
    # Fast_RCNN
    ###################################
    ROI_SIZE = 7
    FAST_RCNN_NMS_IOU_THRESHOLD = 0.3
    FINAL_SCORE_THRESHOLD = 0.7
    FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
    FAST_RCNN_MINIBATCH_SIZE = 200
    FAST_RCNN_POSITIVE_RATE = 0.33
    DETECTION_MAX_INSTANCES = 200

    def __init__(self):
        
        self.NUM_GPUS = len(self.GPU_GROUPS)
        self.BATCH_SIZE = self.NUM_GPUS * self.PER_GPU_IMAGE
        self.BOUNDARY =  [self.NUM_ITEM_DATASET * i // self.BATCH_SIZE for i in self.EPOCH_BOUNDARY] 
        self.SAVE_EVERY_N_STEP= int(self.NUM_ITEM_DATASET/self.BATCH_SIZE)
        # (h ,w)
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.TARGET_SIDE / stride)),
              int(math.ceil(self.TARGET_SIDE / stride))]
             for stride in self.BACKBONE_STRIDES])
