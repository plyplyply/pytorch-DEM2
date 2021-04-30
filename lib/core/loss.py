# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Yiheng Peng (180910334@mail.dhu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp.cuda()
    return inp

class HeatmapLoss(nn.Module):
    '''
    single scale for oo's sake
    keep the dim of the batch
    '''
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):

        assert pred.size() == gt.size()
        loss = ((pred - gt) ** 2) * mask[:, None, :, :].expand_as(pred)
        loss[:, -1, :, :] *= 5 # enhance
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)

        return loss


class OffsetLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.SmoothL1Loss().cuda()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()
        loss = self.criterion(pred, gt)

        return loss

class AELoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """
        global associative embedding loss for one image
        pull same person
        push different
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp) # =>[[],[],[]..]
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)
            #
            #attention ! -1
            #mean: 1/u
            #pull:torch([])

        num_tags = len(tags)
        if num_tags == 0:
            # push, pull
            return make_input(torch.zeros(1).float()), \
                make_input(torch.zeros(1).float())
            return make_input(torch.zeros(1).float()), \
                pull/(num_tags)

        tags = torch.stack(tags)
        size = (num_tags, num_tags) # (PERSON,PERSON)
        A = tags.expand(*size)
        B = A.permute(1, 0) # exchange

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')

        return push/((num_tags - 1) * num_tags) * 0.5, \
            pull/(num_tags) # num_tags == num_people

    def forward(self, tags, joints):
        """
        accumulate tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


class JointsMSELoss(nn.Module):

    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # init check

        self.num_joints = cfg.MODEL.NUM_JOINTS + 1
        self.num_stages = cfg.LOSS.NUM_STAGES
        self.rjg_tag = cfg.DATASET.RJG_TAG

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in cfg.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.heatmaps_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        self.ae_loss = \
            nn.ModuleList(
                [
                    AELoss(cfg.LOSS.AE_LOSS_TYPE) if with_ae_loss else None
                    for with_ae_loss in cfg.LOSS.WITH_AE_LOSS
                ]
            )
        self.push_loss_factor = cfg.LOSS.PUSH_LOSS_FACTOR
        self.pull_loss_factor = cfg.LOSS.PULL_LOSS_FACTOR

        self.offsets_loss = \
            nn.ModuleList(
                [
                    OffsetLoss()
                    if with_heatmaps_loss else None # None ensures that there is no index problem
                    for with_heatmaps_loss in cfg.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.offsets_loss_factor = cfg.LOSS.OFFSETS_LOSS_FACTOR

    def forward(self, outputs, heatmaps, masks, joints, offsets):
        # forward check
        # [joints.copy() for _ in range(self.num_scales)]
        # self._forward_check(outputs, heatmaps, masks, joints)

        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        offsets_losses = []

        tags_pred = outputs[2][:, self.num_joints:]
        batch_size = tags_pred.size()[0]
        tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)
        # 17*height*width

        push_loss, pull_loss = self.ae_loss[0](
            tags_pred, joints[0]
        )
        push_loss = push_loss * self.push_loss_factor[0]
        pull_loss = pull_loss * self.pull_loss_factor[0]

        push_losses.append(push_loss)
        pull_losses.append(pull_loss)

        offset_feat = self.num_joints
        for idx in range(self.num_stages):
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx + 2][:, :offset_feat]

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmaps[idx], masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)

                offsets_pred = outputs[idx]

                offsets_loss = self.offsets_loss[idx](
                    offsets_pred * masks[idx][:, None, :, :].expand_as(offsets_pred),
                    offsets[idx] * masks[idx][:, None, :, :].expand_as(offsets_pred)
                )
                # mask[:, None, :, :].expand_as(pred)

                offsets_loss = offsets_loss * self.offsets_loss_factor[idx]
                offsets_losses.append(offsets_loss)
            else:
                heatmaps_losses.append(None)
                offsets_losses.append(None)

        return heatmaps_losses, push_losses, pull_losses, offsets_losses

    def _init_check(self, cfg):
        assert isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)), \
            'LOSS.WITH_HEATMAPS_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)), \
            'LOSS.HEATMAPS_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.WITH_AE_LOSS, (list, tuple)), \
            'LOSS.WITH_AE_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == len(cfg.LOSS.HEATMAPS_LOSS_FACTOR), \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.HEATMAPS_LOSS_FACTOR should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), len(cfg.LOSS.HEATMAPS_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_AE_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_AE_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PUSH_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PUSH_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PUSH_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PULL_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PULL_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PULL_LOSS_FACTOR))

    def _forward_check(self, outputs, heatmaps, masks, joints):
        assert isinstance(outputs, list), \
            'outputs should be a list, got {} instead.'.format(type(outputs))
        assert isinstance(heatmaps, list), \
            'heatmaps should be a list, got {} instead.'.format(type(heatmaps))
        assert isinstance(masks, list), \
            'masks should be a list, got {} instead.'.format(type(masks))
        assert isinstance(joints, list), \
            'joints should be a list, got {} instead.'.format(type(joints))
        assert len(outputs) == self.num_stages, \
            'len(outputs) and num_stages should been same, got {} vs {}.'.format(len(outputs), self.num_stages)
        assert len(outputs) == len(heatmaps), \
            'outputs and heatmaps should have same length, got {} vs {}.'.format(len(outputs), len(heatmaps))
        assert len(outputs) == len(masks), \
            'outputs and masks should have same length, got {} vs {}.'.format(len(outputs), len(masks))
        assert len(outputs) == len(joints), \
            'outputs and joints should have same length, got {} vs {}.'.format(len(outputs), len(joints))
        assert len(outputs) == len(self.heatmaps_loss), \
            'outputs and heatmaps_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.heatmaps_loss))
        assert len(outputs) == len(self.ae_loss), \
            'outputs and ae_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.ae_loss))


def test_ae_loss():
    import numpy as np
    t = torch.tensor(
        np.arange(0, 32).reshape(1, 2, 4, 4).astype(np.float)*0.1,
        requires_grad=True
    )
    t.register_hook(lambda x: print('t', x))

    ae_loss = AELoss(loss_type='exp')

    joints = np.zeros((2, 2, 2))
    joints[0, 0] = (3, 1)
    joints[1, 0] = (10, 1)
    joints[0, 1] = (22, 1)
    joints[1, 1] = (30, 1)
    joints = torch.LongTensor(joints)
    joints = joints.view(1, 2, 2, 2)

    t = t.contiguous().view(1, -1, 1)
    l = ae_loss(t, joints)

    print(l)

class LossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.heatmaps_loss = None
        self.ae_loss = None
        self.heatmaps_loss_factor = 1.0
        self.push_loss_factor = 1.0
        self.pull_loss_factor = 1.0

        if cfg.LOSS.WITH_HEATMAPS_LOSS:
            self.heatmaps_loss = HeatmapLoss()
            self.heatmaps_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR
        if cfg.LOSS.WITH_AE_LOSS:
            self.ae_loss = AELoss(cfg.LOSS.AE_LOSS_TYPE)
            self.push_loss_factor = cfg.LOSS.PUSH_LOSS_FACTOR
            self.pull_loss_factor = cfg.LOSS.PULL_LOSS_FACTOR

        if not self.heatmaps_loss and not self.ae_loss:
            logger.error('At least enable one loss!')

    def forward(self, outputs, heatmaps, masks, joints):

        heatmaps_pred = outputs[:, :self.num_joints]
        tags_pred = outputs[:, self.num_joints:]

        heatmaps_loss = None
        push_loss = None
        pull_loss = None

        if self.heatmaps_loss is not None:
            heatmaps_loss = self.heatmaps_loss(heatmaps_pred, heatmaps, masks)
            heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor

        if self.ae_loss is not None:
            batch_size = tags_pred.size()[0]
            tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)#展品了

            push_loss, pull_loss = self.ae_loss(tags_pred, joints)
            push_loss = push_loss * self.push_loss_factor
            pull_loss = pull_loss * self.pull_loss_factor

        return [heatmaps_loss], [push_loss], [pull_loss]


if __name__ == '__main__':
    test_ae_loss()
