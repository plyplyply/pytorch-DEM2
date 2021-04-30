# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Some code is from https://github.com/princeton-vl/pose-ae-train/blob/454d4ba113bbb9775d4dc259ef5e6c07c2ceed54/utils/group.py
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Yiheng Peng (180910334@mail.dhu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from munkres import Munkres
import numpy as np
import torch

#tags.append(tag[i, y, x]) = tags.append(tag[i, y, x])


def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp

class Params(object):
    def __init__(self, cfg):
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE

        self.detection_threshold = cfg.TEST.DETECTION_THRESHOLD
        self.tag_threshold = cfg.TEST.TAG_THRESHOLD
        self.use_detection_val = cfg.TEST.USE_DETECTION_VAL
        self.ignore_too_much = cfg.TEST.IGNORE_TOO_MUCH


'''
        if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
            self.num_joints -= 1
        if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
            self.joint_order = [
                i-1 for i in [18, 1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            self.joint_order = [
                i-1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
'''

with_hier_refine_val = False

def hierarchical_pool(heatmap):
    '''
    this function is copied from DEKR
    :param heatmap:
    :return:
    '''
    pool1 = torch.nn.MaxPool2d(3, 1, 1)
    pool2 = torch.nn.MaxPool2d(5, 1, 2)
    pool3 = torch.nn.MaxPool2d(7, 1, 3)
    map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
    if map_size > 300:
        maxm = pool3(heatmap[:, :, :])
    elif map_size > 200:
        maxm = pool2(heatmap[:, :, :])
    else:
        maxm = pool1(heatmap[:, :, :])

    return maxm


class HeatmapParser(object):
    def __init__(self, cfg):
        self.params = Params(cfg)
        self.tag_per_joint = cfg.MODEL.TAG_PER_JOINT
        self.pool = torch.nn.MaxPool2d(
            cfg.TEST.NMS_KERNEL, 1, cfg.TEST.NMS_PADDING
        )
        self.detection_threshold = 0.1
        self.rjg = cfg.DATASET.RJG_TAG
        self.ljg_4 = cfg.DATASET.LJG_WITH_HIER
        self.ljg_2 = cfg.DATASET.LJG_CENTER_ONLY
        self.flip = cfg.TEST.FLIP_TEST
        self.diff_thre = 70.
        self.mrm_thre = 0.75

    def match_by_de(self, dict, params):
        '''
        tag_val(1，num_joints,topk,2)
        location(1,num_joints,topk,2)
        heatmap_val(1,num_joints,topk)
        '''

        assert isinstance(params, Params), 'params should be class Params()'
        tag_t = dict['tag_t']         # [len(rjg)*(thre,2)]
        off_2, off_4 = dict['off_t']  # [len(ljg_2or4)*(thre,2or4)]
        loc_t = dict['loc_t']         # [18*(thre,2)] vs location(1,num_joints,topk,2)
        val_t = dict['val_t']         # [18*(thre)] vs heatmap_val(1,num_joints,topk)

        default_ = np.zeros((params.num_joints + 1, 3 + 2))
        # (30,x+y+val+4+tag+tag_flip)30*5
        joint_dict = {}
        tag_dict = {}
        center2_key = []
        other2_key = [[],[],[],[]]
        # store the keys of shouler l,shoulder r, hip l, hip r

        center_array = loc_t[-1]
        for i,person_center_tag in enumerate(tag_t[-1]):
            key = person_center_tag[0]
            joint_dict.setdefault(key, np.copy(default_))[-1] = np.array([loc_t[-1][i][0], loc_t[-1][i][1], val_t[-1][i], key, person_center_tag[1]])
            tag_dict[key] = [person_center_tag]
            center2_key.append(key)

        # center2_key = np.array(center2_key)

        if center2_key == []:
            return np.array([])

        for idx_tag, idx in enumerate(self.rjg):
            tags = tag_t[idx_tag]
            joints = np.concatenate(
                (loc_t[idx], val_t[idx][:, None], tags), 1
            )

            if joints.shape[0] == 0:
                continue

            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            # print('mean')
            # print(np.mean(tag_dict[0], axis=0))

            if params.ignore_too_much \
                    and len(grouped_keys) == params.max_num_people:
                continue

            # print('joints:', joints.shape)
            # print('tags:', np.array(grouped_tags))

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]

            # joints：（thre，1，tag*2） tags：（1，thre，tag*2）
            # print('j_tag',joints[:, None, 3:])

            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (
                        diff_normed,
                        np.zeros((num_added, num_added - num_grouped)) + 1e10
                    ),
                    axis=1
                )


            pairs = py_max_match(diff_normed)
            for row, col in pairs:
                if (
                        row < num_added
                        and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold
                ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])

                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]
                # other2_key[idx_tag].setdefault(row, key)
                if idx_tag < 4:
                    other2_key[idx_tag].append(key)

            # print('joints_cur',joints.shape)
            # print('keys after once',len(joint_dict.keys()))

        for idx_off, idx in enumerate(self.ljg_2):
            center_key = center2_key
            center_t = center_array.copy()
            center_towards = loc_t[idx] + off_2[idx_off]

            joints = np.concatenate(
                (loc_t[idx], val_t[idx][:, None], off_2[idx_off]), 1
            )

            while(center_t.shape[0] > 0 and center_towards.shape[0] > 0):

                diff = np.sum(np.square(center_towards[:, None, :] - center_t[None, :, :]), axis=-1)
                connection = np.where(diff == diff.min())
                if diff[connection[0][0], connection[1][0]] > self.diff_thre ** 2:
                    break
                key = center_key[connection[1][0]]
                joint_dict[key][idx] = joints[connection[0][0]]

                center_key = np.delete(center_key, connection[1][0], axis=0)
                center_t = np.delete(center_t, connection[1][0], axis=0)
                center_towards = np.delete(center_towards, connection[0][0], axis=0)
                joints = np.delete(joints, connection[0][0], axis=0)

        hier_pre = [[], []]
        for idx_off, idx in enumerate(self.ljg_4):

            ident = 0 + idx_off % 2 if idx_off < 4 else 2 + idx_off % 2
            hier2_key = other2_key[ident]
            hier_t = loc_t[idx - 2].copy() if idx_off in [0, 1, 4, 5] else np.array(hier_pre[0 + idx_off % 2])

            hier_towards = loc_t[idx] + off_4[idx_off][:, :2]
            center_towards = loc_t[idx] + off_4[idx_off][:,2:]
            '''
            center_key = center2_key.copy()
            center_t = center_array.copy()
            '''
            inter_key = []
            inter_loc = []
            # maintain the relationship between loc with new order and their keys
            # tag_used = []

            joints = np.concatenate(
                (loc_t[idx], val_t[idx][:, None], off_4[idx_off][:, :2]), 1
            )

            while(hier_t.shape[0] > 0 and hier_towards.shape[0] > 0):
                diff = np.sum(np.square(hier_towards[:, None, :] - hier_t[None, :, :]), axis=-1)
                connection = np.where(diff == diff.min())
                key = hier2_key[connection[1][0]]
                joint_dict[key][idx] = joints[connection[0][0]]
                inter_key.append(key)
                inter_loc.append(joints[connection[0][0]][:2])


                hier2_key = np.delete(hier2_key, connection[1][0], axis=0)
                hier_t = np.delete(hier_t, connection[1][0], axis=0)
                hier_towards = np.delete(hier_towards, connection[0][0], axis=0)
                joints = np.delete(joints, connection[0][0], axis=0)
                center_towards = np.delete(center_towards, connection[0][0], axis=0)

                '''
                center_c = np.where(np.array(center_key) == key)
                center_key = np.delete(center_key, center_c[0], axis=0)
                center_t = np.delete(center_t, center_c[0], axis=0)
                '''
            left_key = list(set(joint_dict.keys()) ^ set(inter_key))

            if not len(left_key)==0:
                left_key_li = []
                for key in left_key:
                    left_key_li.append(joint_dict[key][-1][:2])
                center_t = np.stack(left_key_li, axis = 0)

                while (center_t.shape[0] > 0 and center_towards.shape[0] > 0):
                    diff = np.sum(np.square(center_towards[:, None, :] - center_t[None, :, :]), axis=-1)
                    connection = np.where(diff == diff.min())
                    if diff[connection[0][0],connection[1][0]] > self.diff_thre ** 2:
                        break
                    key = left_key[connection[1][0]]
                    joint_dict[key][idx] = joints[connection[0][0]]

                    if with_hier_refine_val: # and joints[connection[0][0]][2] >= 0.75:
                        joint_dict[key][idx - 2][:2] = joints[connection[0][0]][:2] + joints[connection[0][0]][3:]
                        joint_dict[key][idx - 2][2] = self.params.detection_threshold

                    inter_key.append(key)
                    inter_loc.append(joints[connection[0][0]][:2])

                    left_key = np.delete(left_key, connection[1][0], axis=0)
                    center_t = np.delete(center_t, connection[1][0], axis=0)
                    center_towards = np.delete(center_towards, connection[0][0], axis=0)
                    joints = np.delete(joints, connection[0][0], axis=0)

            hier_pre[0 + idx_off % 2] = inter_loc
            other2_key[ident] = inter_key
        ans = []
        for i in joint_dict:
            if np.sum(joint_dict[i][:-1, 2]) > 0:
                ans.append(joint_dict[i][:-1, :])
        ans = np.array(ans).astype(np.float32)  # num_people * 17 * 5
        return ans

    def nms(self, det):
        # maxm = self.pool(det)
        maxm = torch.cat([self.pool(det[:-1]), hierarchical_pool(torch.unsqueeze(det[-1], dim = 0))], dim=0)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def nms_top_k_thre(self, tag, off, det):
        # det = torch.Tensor(det, requires_grad=False)
        # tag = torch.Tensor(tag, requires_grad=False)
        #
        det = self.nms(det) # nms

        num_joints = det.size(0)
        h = det.size(1)
        w = det.size(2)
        z = ( h * h  + w * w ) ** 0.5

        det = det.view(num_joints, -1) # h*w
        tag = tag.view(tag.size(0), h * w, -1) # len(rjg)+1...
        off = off.view(off.size(0), -1) * z # 42or36*h*w

        val_k, ind = det.topk(self.params.max_num_people, dim = 1) # （17，30）

        tag_count = 0
        off_count = 0
        tag_t = []
        off_2 = []
        off_4 = []
        val_t = []
        loc_t = []
        x = ind % w
        y =torch.true_divide(ind, w).long()
        # ind_k = torch.stack((x, y), dim=2)
        mask1 = val_k[:-1] > self.detection_threshold
        mask2 = val_k[-1] > 0.01
        mask = torch.cat((mask1, torch.unsqueeze(mask2, 0)), 0)
        # mask = val_k > self.detection_threshold


        for j in range(det.size(0)): # traverse joints
            mask_j = mask[j, :]

            val_t.append(val_k[j, :][mask_j].cpu().numpy())
            loc_t.append \
            (
                torch.stack(
                    [
                        x[j, :][mask_j],
                        y[j, :][mask_j]
                    ], dim = 1).cpu().numpy()
            )

            if j in self.rjg or j == det.size(0) - 1:
                #mask_t = torch.stack([mask_j, mask_j], dim = 1)

                ii = 1 if self.flip else 0
                tag_t.append \
                (
                    torch.stack(
                    [
                        torch.gather(tag[tag_count, :, 0], 0, ind[j, :])[mask_j], # same ind and mask for i
                        torch.gather(tag[tag_count, :, ii], 0, ind[j, :])[mask_j]
                    ],
                        dim = 1
                    ).cpu().numpy()
                )
                tag_count += 1

            else:
                order = [1, 0] if j in self.ljg_2 else [1, 0, 3, 2]
                # switch the order of (offy,offx) to (offx,offy)
                # switch the order of hier,basic to basic,hier
                # mask_j = torch.stack([mask_j for _ in range(length)], dim = 1)

                inter = \
                [
                    torch.gather(off[off_count + d, :], 0, ind[j, :])[mask_j]
                    for d in order
                ]

                if len(order) == 2:
                    off_2.append \
                    (
                        torch.stack(inter, dim = 1).cpu().numpy()
                        # （30，2or4：center xy，limb xy）
                    )
                else:
                    off_4.append \
                    (
                        torch.stack(inter, dim = 1).cpu().numpy()
                        # （30，2or4：center xy，limb xy）
                    )
                off_count += len(order)


        ans = {
            'tag_t': tag_t,        # [len(rjg)*(thre,2)]
            'off_t': (off_2,off_4),# [len(ljg)*(thre,2or4)]
            'loc_t': loc_t,        # [18*(thre,2)] vs location(1,num_joints,topk,2)
            'val_t': val_t         # [18*(thre)] vs heatmap_val(1,num_joints,topk)
        }

        return ans


    def adjust(self, ans, det):

        for joint_id, joint in enumerate(ans):
            if joint[2] > 0:
                tmp = det[joint_id]
                y, x = joint[0:2]
                xx, yy = int(x), int(y)
                xx = tmp.shape[0] - 1 if xx > tmp.shape[0] - 1 else xx
                xx = 0 if xx < 0 else xx
                yy = tmp.shape[1] - 1 if yy > tmp.shape[1] - 1 else yy
                yy = 0 if yy < 0 else yy
                # print(batch_id, joint_id, det[batch_id].shape)

                # hm
                # (1, 17, 512, 768)
                if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
                    y += 0.25
                else:
                    y -= 0.25

                if tmp[min(xx+1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
                    x += 0.25
                else:
                    x -= 0.25
                ans[joint_id, 0:2] = (y + 0.5, x + 0.5)
        return ans

    def refine_rjg_only(self, keypoints, det, tag):
        """
        Given initial keypoint predictions, we identify missing joints
        :param det: numpy.ndarray of size (17, 128, 128)
        :param tag: numpy.ndarray of size (17, 128, 128) if not flip
        :param keypoints: numpy.ndarray of size (17, 4) if not flip, last dim is (x, y, det score, tag score)
        :return:
        important refinement!!!
        """
        # tag ndarray shape: (17, 640, 1024, 2)
        tags = []
        for i in self.rjg:
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(np.int32)
                tags.append(tag[i, y, x])

        if tags == []:
            return keypoints

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0) # (1,)or(2,)
        ans = []

        for i in self.rjg:
            tmp = det[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
            tmp2 = tmp - np.round(tt)

            # find maximum position
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            xx = x
            yy = y
            # detection score at maximum position
            val = tmp[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i_tag,i in enumerate(self.rjg):
                # add keypoint if it is not detected
                if ans[i_tag, 2] > 0 and keypoints[i, 2] == 0:
                # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i_tag, :2]
                    keypoints[i, 2] = ans[i_tag, 2]

        return keypoints

    def refine(self, keypoints, det, tag):
        """
        Given initial keypoint predictions, we identify missing joints
        :param det: numpy.ndarray of size (17, 128, 128)
        :param tag: numpy.ndarray of size (17, 128, 128) if not flip
        :param keypoints: numpy.ndarray of size (17, 4) if not flip, the last dim is (x, y, det score, tag score)
        :return:
        """
        if len(tag.shape) == 3:
            # for ensurance's sake
            tag = tag[:, :, :, None]

        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(np.int32)
                tmp = det[i, :, :]
                x = tmp.shape[1] - 1 if x > tmp.shape[1] - 1 else x
                x = 0 if x < 0 else x
                y = tmp.shape[0] - 1 if y > tmp.shape[0] - 1 else y
                y = 0 if y < 0 else y
                # print(x,y,tmp.shape[0],tmp.shape[1])
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        for i in range(keypoints.shape[0]): # if to change，here to be reduced
            # score of joints i at all position
            tmp = det[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
            tmp2 = tmp - np.round(tt)

            # find maximum position
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            xx = x
            yy = y
            # detection score at maximum position
            val = tmp[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(keypoints.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]

        return keypoints

    def mutual_refine_machine(self, ans):
        ans_r = ans.copy()
        '''
        for j in self.ljg_2:
            if ans[-1, 2] <= self.mrm_thre:
                break
            if ans[-1, 2] > self.mrm_thre and ans[-1, 2] > ans[j, 2] and ans[j, 2] > 0:
                Q = ans[-1, 2] + ans[j, 2]
                ans_r[j, :2] = ans[j, 2] / Q * ans[j, :2] + ans[-1, 2] / Q * (ans[-1, :2] - ans[j, 3:])
        '''
        for off_idx, j in enumerate(self.ljg_4):
            if off_idx not in [0, 1, 4, 5]:
                if ans[j - 2, 2] > self.mrm_thre and ans[j - 2, 2] > ans[j, 2] and ans[j, 2] > 0:
                    Q = ans[j - 2, 2] + ans[j, 2]
                    ans_r[j, :2] = ans[j, 2] / Q * ans[j, :2] + ans[j - 2, 2] / Q * (
                            ans[j - 2, :2] - ans[j, 3:])
            else:
                Q = ans[j, 2]
                if not Q > 0:
                    continue
                for ii in [j - 2, j + 2]:
                    if ans[ii, 2] > self.mrm_thre and ans[ii, 2] > ans[j, 2]:
                        Q += ans[ii, 2]
                if Q == ans[j, 2]:
                    continue
                else:
                    sum = 0
                    ii = j - 2
                    if ans[ii, 2] > self.mrm_thre and ans[ii, 2] > ans[j, 2]:
                        sum += ans[ii, 2] / Q * (ans[ii, :2] - ans[j, 3:])
                    ii = j + 2
                    if ans[ii, 2] > self.mrm_thre and ans[ii, 2] > ans[j, 2]:
                        sum += ans[ii, 2] / Q * (ans[ii, :2] + ans[ii, 3:])

                    ans_r[j, :2] = ans[j, 2] / Q * ans[j, :2] + sum

        for off_idx, j in enumerate(self.rjg):
            Q = ans[j, 2]
            if not Q > 0:
                continue
            for ii in [j + 2]:
                if ans[ii, 2] > self.mrm_thre and ans[ii, 2] > ans[j, 2]:
                    Q += ans[ii, 2]
            if Q == ans[j, 2]:
                continue
            else:
                sum = 0
                ii = j + 2
                sum += ans[ii, 2] / Q * (ans[ii, :2] + ans[ii, 3:])
                ans_r[j, :2] = ans[j, 2] / Q * ans[j, :2] + sum
        return ans_r

    def parse(self, det, tag, off, adjust=True, refine=True):
        ans = self.match_by_de(self.nms_top_k_thre(torch.squeeze(tag[:, self.rjg + [-1]]), torch.squeeze(off), torch.squeeze(det)), self.params)
        scores = [i[:, 2].mean() for i in ans]
        if refine:
            # for every detected person
            for i in range(len(ans)):
                det_numpy = det[0].cpu().numpy()
                tag_numpy = tag[0].cpu().numpy()
                '''
                    in parse
                    torch.Size([1, 17, 512, 768])
                    torch.Size([1, 17, 512, 768, 2])
                '''
                # ans[i] = self.mutual_refine_machine(ans[i])
                ans[i] = self.adjust(ans[i], det_numpy)
                ans[i] = self.refine(ans[i], det_numpy, tag_numpy)
            # if ans.shape[0] > 0:
            # ans = ans[:, :-1, :]
            # do refine
            # np.array([array,array])
        ans = [ans]
        # print(ans)
        return ans, scores
