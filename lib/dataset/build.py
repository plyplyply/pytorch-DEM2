# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .COCODataset import CocoDataset as coco
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import ScaleAwareHeatmapGenerator
from .target_generators import JointsGenerator
from .target_generators import OffsetGenerator
'''
DATASET:
  BASE_SIGMA: 2.0
  BASE_SIZE: 256.0
  DATASET: coco_kpt
  DATASET_TEST: coco
  DATA_FORMAT: jpg
  FLIP: 0.5
  INPUT_SIZE: 512
  INT_SIGMA: False
  MAX_NUM_PEOPLE: 30
  MAX_ROTATION: 30
  MAX_SCALE: 1.5
  MAX_TRANSLATE: 40
  MIN_SCALE: 0.75
  NUM_JOINTS: 17
  OUTPUT_SIZE: [128, 256]
  ROOT: data/coco
  SCALE_AWARE_SIGMA: False
  SCALE_TYPE: short
  SIGMA: 2
  TEST: val2017
  TRAIN: train2017
  WITH_CENTER: False
'''

def build_dataset(cfg, is_train):

    transforms = build_transforms(cfg, is_train)

    if cfg.DATASET.SCALE_AWARE_SIGMA:
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.DATASET.NUM_JOINTS, cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    joints_generator = [
        JointsGenerator(
            cfg.DATASET.RJG_TAG,
            cfg.DATASET.MAX_NUM_PEOPLE,
            cfg.DATASET.NUM_JOINTS,
            output_size,
            cfg.MODEL.TAG_PER_JOINT
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    offset_generator = [
        OffsetGenerator(
            cfg,
            output_size
        )
        for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
    #dataset_name='data\coco\annotations\person_keypoints_train2017.json'

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        dataset_name,
        is_train,
        heatmap_generator,
        joints_generator,
        offset_generator,
        transforms
    )

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):

    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, is_train)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,

    )

    return data_loader


def make_test_dataloader(cfg):
    transforms = None
    dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST,
        cfg.DATASET.DATA_FORMAT,
        transforms
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
