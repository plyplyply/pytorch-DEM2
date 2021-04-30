# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Yiheng Peng (180910334@mail.dhu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
import numpy as np
import _init_paths
import models
import time

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


    model1 = eval('models.'+ 'pose_higher_hrnet_de' +'.get_pose_net')(
        cfg, is_train = False
    )

    dump_input = torch.rand(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    #logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:
        model1 = network_to_half(model1)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model1.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        ) # 'model_best.pth.tar'
        logger.info('=> loading model1 from {}'.format(model_state_file))
        # model1.load_state_dict(torch.load(model_state_file))
        loading_from_mm = False
        if loading_from_mm:
            state_dict = torch.load(model_state_file)
            # create new OrderedDict that does not contain 'module.'
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            count = 0
            for k1, v1 in state_dict.items():
                if (count == 0):
                    count += 1
                    continue
                # print(v1)
                for k, v in v1.items():
                    i = 8 if 'backbone' in k else 13
                    name = k[i + 1:]  # remove 'module.'
                    new_state_dict[name] = v
            model1.load_state_dict(new_state_dict)
        else:
            state_dict = torch.load(model_state_file)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                i = -1
                name = k[i + 1:]
                new_state_dict[name] = v
            model1.load_state_dict(new_state_dict)

        '''
        model_state_file = os.path.join(
            final_output_dir, 'pose_higher_hrnet_de_head_w32_512.pth'
        )  # 'model_best.pth.tar'
        logger.info('=> loading model2 from {}'.format(model_state_file))
        # model1.load_state_dict(torch.load(model_state_file))
        model2.load_state_dict(torch.load(model_state_file))
        '''

    model1 = torch.nn.DataParallel(model1, device_ids=cfg.GPUS).cuda()
    model1.eval()

    data_loader, test_dataset = make_test_dataloader(cfg)

    if cfg.MODEL.NAME1 == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    #
                )
            ]
        )

    parser = HeatmapParser(cfg)
    all_preds = []
    all_scores = []

    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
    # bar printing

    with torch.no_grad():
        for i, (images, annos) in enumerate(data_loader):
            
            assert 1 == images.size(0), 'Test batch size should be 1'
            print("picture{0}:".format(i))
            image = images[0].cpu().numpy()
            # size at scale 1.0
            base_size, center, scale = get_multi_scale_size(
                image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
            )
            # current_scale：1.0
            # return (w_resized, h_resized), center, np.array([scale_w, scale_h])

            with torch.no_grad():
                final_heatmaps = None
                final_offsets = None
                tags_list = []
                a = time.time()
                for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                    input_size = cfg.DATASET.INPUT_SIZE
                    image_resized, center, scale = resize_align_multi_scale(
                        image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                    )

                    image_resized = transforms(image_resized) # Tensorfy+normalize
                    image_resized = image_resized.unsqueeze(0).cuda() # set batch_size 1 + to cuda
                    # print('model1.device',model1.device)
                    # print('image_resized.device',image_resized.device)
                    # x = model1(image_resized)

                    heatmaps, tags ,offsets = get_multi_stage_outputs(
                        cfg, model1, image_resized, cfg.TEST.FLIP_TEST,
                        cfg.TEST.PROJECT2IMAGE, base_size
                    )

                    final_heatmaps, tags_list, final_offsets = aggregate_results(
                        cfg, s, final_heatmaps, tags_list, final_offsets, heatmaps, tags, offsets
                    )

                final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
                # (batch,channel,height,width)
                tags = torch.cat(tags_list, dim=4)
                final_offsets = final_offsets / float(len(cfg.TEST.SCALE_FACTOR))
                # (batch,channel,height,width)

                grouped, scores = parser.parse(
                    final_heatmaps, tags, final_offsets, cfg.TEST.ADJUST, cfg.TEST.REFINE
                )

                final_results = get_final_preds(
                    grouped, center, scale,
                    [final_heatmaps.size(3), final_heatmaps.size(2)]
                )
                b = time.time()
                print('inference with flip and scale:', b - a)

            if cfg.TEST.LOG_PROGRESS:
                pbar.update()

            if i % 1 == 0:#cfg.PRINT_FREQ == 0:
                prefix = '{}_{}'.format(os.path.join(final_output_dir, 'result_valid'), i)
                # logger.info('=> write {}'.format(prefix))
                save_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset=test_dataset.name)
                # save_debug_images(cfg, image_resized, None, None, outputs, prefix)

            all_preds.append(final_results)
            all_scores.append(scores)

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()

    name_values, _ = test_dataset.evaluate(
        cfg, all_preds, all_scores, final_output_dir
    )

    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, cfg.MODEL.NAME1)
    else:
        _print_name_value(logger, name_values, cfg.MODEL.NAME1)


if __name__ == '__main__':
    main()
