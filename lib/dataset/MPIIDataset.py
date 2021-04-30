from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path
import cv2
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset
from utils import zipreader

import matplotlib
matplotlib.use('Agg')
import torch.utils.data as data
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import cv2
import numpy as np
import os
import os.path
import json
import random
import time

# Use opencv to load image
def opencv_loader(path):
    return cv2.imread(path, 1)


# MPII Multi-Person dataset
class MPI_Dataset(data.Dataset):
    def __init__(self,
                 cfg,
                 root,
                 train_file,
                 heatmap_generator,
                 joints_generator,
                 offset_generator,
                 data_aug,
                 joint_trans,
                 transform=None,
                 target_transform=None,
                 loader=opencv_loader, \
                 stride=4, \
                 sigma=7, \
                 crop_size=256, \
                 target_dist=1.171, scale_min=0.7, scale_max=1.3, \
                 max_rotate_degree=40, \
                 max_center_trans=40, \
                 flip_prob=0.5, \
                 is_visualization=False):

        # Load training json file
        print('Loading training json file: {0}...'.format(train_file))
        train_list = []
        with open(train_file) as data_file:
            data_this = json.load(data_file)
            data_this = data_this['root']
            train_list = train_list + data_this
        print('Finished loading training json file')

        # Hyper-parameters
        self.root = root
        self.train_list = train_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.stride = stride
        self.sigma = sigma
        self.crop_size = crop_size
        self.target_dist = target_dist
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.max_rotate_degree = max_rotate_degree
        self.max_center_trans = max_center_trans
        self.flip_prob = flip_prob

        # Number of train samples
        self.N_train = len(self.train_list)

        # Visualization or not
        self.is_visualization = is_visualization

    def __getitem__(self, index):
        pass
    def __len__(self):
        return self.N_train



