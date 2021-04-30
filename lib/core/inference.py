# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Yiheng Peng (180910334@mail.dhu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch

from dataset.transforms import FLIP_CONFIG


def get_outputs(
        cfg, model, image, with_flip=False,
        project2image=False, size_projected=None
):
    outputs = []
    heatmaps = []
    tags = []

    outputs.append(model(image))
    heatmaps.append(outputs[-1][:, :cfg.DATASET.NUM_JOINTS])
    tags.append(outputs[-1][:, cfg.DATASET.NUM_JOINTS:])

    if with_flip:
        outputs.append(model(torch.flip(image, [3])))
        outputs[-1] = torch.flip(outputs[-1], [3])
        heatmaps.append(outputs[-1][:, :cfg.DATASET.NUM_JOINTS])
        tags.append(outputs[-1][:, cfg.DATASET.NUM_JOINTS:])
        if 'coco' in cfg.DATASET.DATASET:
            dataset_name = 'COCO'
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            dataset_name = 'CROWDPOSE'
        else:
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
        flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER'] \
            if cfg.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]
        heatmaps[-1] = heatmaps[-1][:, flip_index, :, :]
        if cfg.MODEL.TAG_PER_JOINT:
            tags[-1] = tags[-1][:, flip_index, :, :]

    if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
        heatmaps = [hms[:, :-1] for hms in heatmaps]
        tags = [tms[:, :-1] for tms in tags]

    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]

    return outputs, heatmaps, tags


def get_multi_stage_outputs(
        cfg, model, image, with_flip=False,
        project2image=False, size_projected=None
):
    heatmaps = []
    tags = []
    offsets = []
    # outputs = []

    heatmaps_avg = 0
    offsets_avg = 0
    num = 0
    num_off = 0

    outputs = model(image)

    for i, output in enumerate(outputs):

        if len(outputs) > 1 and i != len(outputs) - 1 and i != len(outputs) - 3:
            output = torch.nn.functional.interpolate(output,
                                        size=(outputs[-1].size(2), outputs[-1].size(3)),
                                        mode='bilinear',
                                        align_corners=False
                                        )
        off_feat = cfg.DATASET.NUM_JOINTS + 1
        if i == 2:
            tags.append(output[:, off_feat:]) # quarter to half
        else: # i==1||i==2
            if cfg.LOSS.WITH_HEATMAPS_LOSS[i % 2] and cfg.TEST.WITH_HEATMAPS[i % 2]:
                if  i < 2: #i == 1: #
                    offsets_avg += output # TO UPPER(X,Y),TO CENTER(X,Y)
                    num_off += 1
                else:
                    heatmaps_avg += output[:, :off_feat]
                    num += 1

    if num > 0:
        heatmaps.append(heatmaps_avg / num)
        offsets.append(offsets_avg / num_off)

    if with_flip:

        if 'coco' in cfg.DATASET.DATASET:
            dataset_name = 'COCO'
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            dataset_name = 'CROWDPOSE'
        else:
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)

        flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER']
        # tag_flip_index=FLIP_CONFIG[dataset_name+'_TAG_WITH_CENTER']
        emb_flip_index=FLIP_CONFIG[dataset_name+'_EMB']
        # if cfg.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]

        final_emb_flip_index = []
        for i, idx in enumerate(emb_flip_index):

            if 'coco' in cfg.DATASET.DATASET:  # coco
                if idx < 5:  # face
                    for j in range(2):
                        final_emb_flip_index.append(0 + (idx - 0) * 2 + j)
                elif idx >= 7 and idx <= 10:  # idx >=5:other joints
                    for j in range(4):
                        final_emb_flip_index.append(10 + (idx - 7) * 4 + j)
                else:
                    for j in range(4):
                        final_emb_flip_index.append(26 + (idx - 13) * 4 + j)

            else:  # crowd_pose
                if idx < 6:
                    for j in range(4):
                        final_emb_flip_index.append(0 + (idx - 2) * 4 + j)
                elif idx >= 8 and idx <= 11:
                    for j in range(4):
                        final_emb_flip_index.append(16 + (idx - 8) * 4 + j)
                else:  # face
                    for j in range(2):
                        final_emb_flip_index.append(32 + (idx - 12) * 2 + j)

        heatmaps_avg_f = 0
        offsets_avg_f = 0
        num_f = 0
        num_off_f = 0

        outputs_flip = model(torch.flip(image, [3]))

        #Reverse the order of a n-D tensor along given axis in dims.

        for i, output_f in enumerate(outputs_flip):
            if len(outputs_flip) > 1 and i != len(outputs_flip) - 1 and i != len(outputs) - 3:
                output_f = torch.nn.functional.interpolate(
                    output_f,
                    size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            output_f = torch.flip(output_f, [3]) # interpolation first

            off_feat = cfg.DATASET.NUM_JOINTS + 1
            if i == 2:
                tags.append(output_f[:, off_feat:])
                tags[-1] = tags[-1][:, flip_index, :, :]
            else:
                if cfg.LOSS.WITH_HEATMAPS_LOSS[i % 2] and cfg.TEST.WITH_HEATMAPS[i % 2]:
                    if   i < 2:
                        offsets_avg_f += \
                            output_f[:, final_emb_flip_index, :, :]
                        num_off_f += 1
                    else:
                        heatmaps_avg_f += \
                            output_f[:, :off_feat][:, flip_index, :, :]
                        num_f += 1



        if num_f > 0:
            heatmaps.append(heatmaps_avg_f / num_f)
            offsets.append(offsets_avg_f / num_off_f)

    # if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
        # heatmaps = [hms[:, :-1] for hms in heatmaps]
        # tags = [tms[:, :-1] for tms in tags]#center for last

    if project2image and size_projected: # upsample
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]

        offsets = [
            torch.nn.functional.interpolate(
                oms,
                size=(size_projected[1],size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for oms in offsets
        ]

    return heatmaps, tags, offsets


def aggregate_results(
        cfg, scale_factor, final_heatmaps, tags_list, final_offsets, heatmaps, tags, offsets
):
    # tag
    if scale_factor == 1 or len(cfg.TEST.SCALE_FACTOR) == 1:#艹，只用了scale=1的
        if final_heatmaps is not None and not cfg.TEST.PROJECT2IMAGE:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    # hms
    heatmaps_avg = (heatmaps[0] + heatmaps[1]) / 2.0 if cfg.TEST.FLIP_TEST \
        else heatmaps[0]    # flip

    # emb
    offsets_mid = offsets[0]
    for i in range(offsets[0].size(1)):
        if i % 2 == 0: # delta y
            offsets_mid[:, i, :, :] += offsets[1][:, i, :, :]
        else: # delta x
            offsets_mid[:, i, :, :] -= offsets[1][:, i, :, :]

    offsets_avg = offsets_mid / 2.0 if cfg.TEST.FLIP_TEST \
        else offsets[0]

    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
        # scale
    elif cfg.TEST.PROJECT2IMAGE:
        final_heatmaps += heatmaps_avg
        # scale test
    else:
        final_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )

    if final_offsets is None:
        final_offsets = offsets_avg
        # scale
    elif cfg.TEST.PROJECT2IMAGE:
        final_offsets += offsets_avg
        # scale test
    else:
        final_offsets += torch.nn.functional.interpolate(
            offsets_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )

    return final_heatmaps, tags_list, final_offsets
