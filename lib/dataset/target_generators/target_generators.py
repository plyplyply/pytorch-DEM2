# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Yiheng Peng (180910334@mail.dhu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class ScaleAwareHeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints

    def get_gaussian_kernel(self, sigma):
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        for p in joints:
            sigma = p[0, 3]
            g = self.get_gaussian_kernel(sigma)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], g[a:b, c:d])
        return hms

class JointsGenerator():
    def __init__(self, RJG, max_num_people, num_joints, output_res, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.output_res = output_res
        self.tag_per_joint = tag_per_joint # True
        self.RJG_TAG = RJG

    def __call__(self, joints_origin):
        '''
        stacking by original joint orders
        np.zeros((num_people, self.num_joints, 4))
        '''
        # inter = self.RJG_TAG + [17] len(inter)
        inter = self.num_joints + 1
        joints = joints_origin
        visible_nodes = np.zeros((self.max_num_people, inter, 2))
        output_res = self.output_res
        for i in range(len(joints)):
            tot = 0
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                if pt[2] > 0 and x >= 0 and y >= 0 \
                   and x < self.output_res and y < self.output_res:
                    if self.tag_per_joint:
                        visible_nodes[i][tot] = \
                            (idx * output_res ** 2 + y * output_res + x, 1)
                    else:
                        visible_nodes[i][tot] = \
                            (y * output_res + x, 1)
                    tot += 1
        return visible_nodes

class HeatmapGenerator():
    '''
    generate target map by HeatmapGenerator
    '''
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res # a list
        self.num_joints = num_joints + 1
        if sigma < 0: # set as 2 in our final version
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6 * sigma + 3 # 15
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1 # 7，7
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

coco_dict = {
        'face' : [0, 1, 2, 3, 4],
        'left_arm' : [5, 7, 9],
        'right_arm' : [6, 8, 10],
        'left_f': [11, 13, 15],
        'right_f': [12, 14, 16]
}
crowdpose_dict = {
        'face' : [12, 13],
        'left_arm' : [0, 2, 4],
        'right_arm' : [1, 3, 5],
        'left_f': [6, 8, 10],
        'right_f': [7, 9, 11]
}

towards = [
    {
    9:7, 7:5, 10:8, 8:6, 15:13, 13:11, 16:14, 14:12
    },
    {
    4:2, 2:0, 5:3, 3:1, 10:8, 8:6, 11:9, 9:7
    }
]

class OffsetGenerator():
    '''
    When it comes to the offsets, stack (offsety, offsetx)
    in the order of the dataset: for example, for coco,
    0th to 4th joints are corresponding to facial points with
    basic offsets (basic_offsety,basic_offsetx) only;
    5th and 6th are torso joints without displacement;
    7th joint is among ljg with both hier-offsets and basic
    displacement:(hier_offsety, hier_offsetx, basic_offsety, basic_offsetx)
    coco:
    {
    0th:channel0,channel1 ((basic_offsety,basic_offsetx))
    1th:2,3
    2th:4,5
    3th:6,7
    4th:8,9
    5th:pass
    6th:pass
    7th:channel10,channel11,channel12,channel13 ((hier_offsety,hier_offsetx,basic_offsety,basic_offsetx))
    8th:14,15,16,17
    9th:18,19,20,21
    10th:22,23,24,25
    11th:pass
    12th:pass
    13th:26,27,28,29
    14th:30,31,32,33
    15th:34,35,36,37
    16th:38,39,40,41
    }

    When the resources are sufficient, we recommend
    disentangling the regression of the offsets,
    which may increase the performance.
    (more computationally expensive)

    '''
    def __init__(self, cfg, output_res, tao=7):
        self.output_res = output_res
        self.num_joints = cfg.DATASET.NUM_JOINTS # with out center
        self.num_channels = 4 * len(cfg.DATASET.LJG_WITH_HIER) + 2 * len(cfg.DATASET.LJG_CENTER_ONLY)
        self.tao = tao
        self.s = 0 if 'coco' in cfg.DATASET.DATASET else 1
        self.towards = towards[0] if 'coco' in cfg.DATASET.DATASET else towards[1]
        self.Z = (2 * ( output_res ** 2 ) ) ** 0.5
        self.rjg = cfg.DATASET.RJG_TAG
        self.ljg_4 = cfg.DATASET.LJG_WITH_HIER
        self.ljg_2 = cfg.DATASET.LJG_CENTER_ONLY

    def __call__(self, joints):
        '''

        '''

        oms = np.zeros((self.num_channels, self.output_res, self.output_res), dtype=np.float32)
        ocs = np.zeros((self.num_channels, self.output_res, self.output_res), dtype=np.float32)

        count = 0 # to be 42 or 36 when the loop is over
        for body_part in range(self.num_joints):
            if body_part in self.rjg:
               continue
            for pidx, person_joint in enumerate(joints[:, body_part]):

                if person_joint[2] > 0:
                    x, y = int(person_joint[0]), int(person_joint[1])  # width,height
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    cc, dd = max(0, x - self.tao), min(self.output_res, x + self.tao + 1)
                    aa, bb = max(0, y - self.tao), min(self.output_res, y + self.tao + 1)
                    cc, dd, aa, bb = int(cc), int(dd), int(aa), int(bb)

                    yy = np.arange(aa, bb)[:, None]  # [[],[]...]=>[[      ],[     ]...]
                    yy = yy + np.zeros(dd - cc)[None, :].astype(np.int32)

                    xx = np.arange(cc, dd)[None, :]  # [[       ]]=>[[     ],[     ]...]
                    xx = xx + np.zeros(bb - aa)[:, None].astype(np.int32)

                    if body_part in self.ljg_2:
                        #center
                        oms[count, aa : bb, cc : dd] += (joints[pidx, -1, 1] - yy) / self.Z
                        oms[count + 1, aa : bb, cc : dd] += (joints[pidx, -1, 0] - xx) / self.Z

                        ocs[count, aa : bb, cc : dd] += 1
                        ocs[count + 1, aa: bb, cc: dd] += 1
                    else:
                        # upper
                        xto = joints[pidx, self.towards[body_part], 0]
                        yto = joints[pidx, self.towards[body_part], 1]

                        if  xto >= 0 and  yto >= 0 and xto < self.output_res and yto < self.output_res:
                            # keep as zero
                            oms[count, aa : bb, cc : dd] = (yto - yy) / self.Z
                            oms[count + 1, aa : bb, cc : dd] = (xto - xx) / self.Z

                            ocs[count, aa: bb, cc: dd] += 1
                            ocs[count + 1, aa: bb, cc: dd] += 1

                        # center. if == true center definitely is visible
                        oms[count + 2, aa : bb, cc : dd] = (joints[pidx, -1, 1] - yy) / self.Z
                        oms[count + 3, aa : bb, cc : dd] = (joints[pidx, -1, 0] - xx) / self.Z

                        for ii in range(2):
                            ocs[count + 2 + ii, aa : bb, cc : dd] += 1

            if body_part in self.ljg_2:
                count += 2
            else:
                count += 4

        ocs[ocs == 0] = 1
        oms /= ocs

        return oms



'''
                    if self.s==0:#coco
                        if i == 0:#face
                            oms[2*body_part, aa : bb, cc : dd] = (joints[pidx,-1,0] - xx ) / self.Z
                            oms[2*body_part+1, aa : bb, cc : dd] = (joints[pidx,-1,1] - yy ) / self.Z
                        else:
                            if j==0:
                                continue
                            elif j==1 or j==2:
                                start = 10 + (body_part - 7) * 4
                            else:
                                start = 26 + (body_part - 13) * 4
                            #upper
                            oms[start, aa: bb, cc: dd] = (joints[pidx, body_part_group[j-1], 0] - xx) / self.Z
                            oms[start + 1, aa: bb, cc: dd] = (joints[pidx, body_part_group[j-1], 1] - yy) / self.Z

                            #center
                            oms[start + 2, aa: bb, cc: dd] = (joints[pidx, -1, 0] - xx) / self.Z
                            oms[start + 3, aa: bb, cc: dd] = (joints[pidx, -1, 1] - yy) / self.Z

                    else:#crowd_pose
                        if i == 0:#face
                            oms[-2 + (-2) * (13-body_part), aa: bb, cc: dd] = (joints[pidx, -1, 0] - xx) / self.Z
                            oms[-1 + (-2) * (13-body_part), aa: bb, cc: dd] = (joints[pidx, -1, 1] - yy) / self.Z
                            #count as y,x minus one more time
                        else:
                            if j==0:
                                continue
                            elif j==1 or j==2:
                                start = 0 + (body_part - 2) * 4
                            else:
                                start = 16 + (body_part - 8) * 4
                            #upper
                            oms[start, aa: bb, cc: dd] = (joints[pidx, body_part_group[j-1], 0] - xx) / self.Z
                            oms[start + 1, aa: bb, cc: dd] = (joints[pidx, body_part_group[j-1], 1] - yy) / self.Z

                            #center
                            oms[start + 2, aa: bb, cc: dd] = (joints[pidx, -1, 0] - xx) / self.Z
                            oms[start + 3, aa: bb, cc: dd] = (joints[pidx, -1, 1] - yy) / self.Z
'''



