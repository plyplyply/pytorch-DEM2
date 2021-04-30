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
import os
import time

import torch


from utils.utils import AverageMeter
from utils.vis import save_debug_images
from torchvision.transforms import functional as F

def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, fp16=False):

    torch.cuda.empty_cache()
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmaps_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    push_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    pull_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    offsets_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]

    # switch to train mode
    model.train()


    end = time.time()
    for i, (images, heatmaps, masks, joints, offsets) in enumerate(data_loader):#gt heatmap
        # measure data loading time
        data_time.update(time.time() - end)

        x = images.cuda()
        #print(images.device)

        # compute output
        outputs = model(x)

        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
        joints = list(map(lambda x: x.cuda(non_blocking=True), joints))
        offsets = list(map(lambda x: x.cuda(non_blocking=True), offsets))

        heatmaps_losses, push_losses, pull_losses, offsets_losses = \
            loss_factory(outputs, heatmaps, masks, joints, offsets)

        loss = 0

        for idx in range(cfg.LOSS.NUM_STAGES):

            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                heatmaps_loss_meter[idx].update(
                    heatmaps_loss.item(), images.size(0)
                )
                loss = loss + heatmaps_loss

                offsets_loss = offsets_losses[idx]
                offsets_loss_meter[idx].update(
                    offsets_loss.item(), images.size(0)
                )
                loss = loss + offsets_loss

                if idx == 0:
                    push_loss = push_losses[idx].mean(dim=0)
                    push_loss_meter[idx].update(
                        push_loss.item(), images.size(0)
                    )
                    loss = loss + push_loss

                    pull_loss = pull_losses[idx].mean(dim=0)
                    pull_loss_meter[idx].update(
                        pull_loss.item(), images.size(0)
                    )
                    loss = loss + pull_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss) # fp 16optimizer.
        else:
            loss.backward()
        optimizer.step()
        #
        #
        #
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{push_loss}{pull_loss}{offsets_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time, # average time
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmaps_loss_meter, 'heatmaps'),
                      push_loss=_get_loss_info(push_loss_meter, 'push'),
                      pull_loss=_get_loss_info(pull_loss_meter, 'pull'),
                      offsets_loss=_get_loss_info(offsets_loss_meter, 'offsets')
                  )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']

            for idx in range(cfg.LOSS.NUM_STAGES):
                writer.add_scalar(
                    'train_stage{}_heatmaps_loss'.format(idx),
                    heatmaps_loss_meter[idx].val, # axis y
                    global_steps # axis x
                )
                writer.add_scalar(
                    'train_stage{}_push_loss'.format(idx),
                    push_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_pull_loss'.format(idx),
                    pull_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_offsets_loss'.format(idx),
                    offsets_loss_meter[idx].val,
                    global_steps
                )

            writer_dict['train_global_steps'] = global_steps + 1

            '''
            root_output_dir = Path(cfg.OUTPUT_DIR)
            
            dataset = cfg.DATASET.DATASET
            dataset = dataset.replace(':', '_')###########
            
            model = cfg.MODEL.NAME
            
            cfg_name = os.path.basename(cfg_name).split('.')[0]
            parser.add_argument('--cfg',
                                help='experiment configure file name',
                                required=True,
                                type=str)
            
            final_output_dir = root_output_dir / dataset / model / cfg_name
            output_dir : final...
            '''

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            #
            num_joints = cfg.MODEL.NUM_JOINTS + 1

            for scale_idx in range(2):

                prefix_scale = prefix + '_output_{}'.format(
                    cfg.DATASET.OUTPUT_SIZE[scale_idx])

                tag_pred = None if scale_idx == 1 else outputs[2][:, num_joints:]

                save_debug_images(
                    cfg, images, heatmaps[scale_idx], masks[scale_idx], offsets[scale_idx],
                    outputs[scale_idx], outputs[scale_idx + 2][:, :num_joints], prefix_scale, tag_pred
                )
                # OUTPUT_SIZE: [160, 320]


def _get_loss_info(loss_meters, loss_name):

    msg = ''

    for i, meter in enumerate(loss_meters): # stage

        msg += 'Stage{i}-{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            i=i, name=loss_name, meter=meter
        )

    return msg
