# train.py
# !/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import pickle
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
from func_3d.inference import test_sam
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader


def main():
    args = cfg.parse_args()

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net.to(dtype=torch.bfloat16)
    if args.pretrain:
        print(args.pretrain)
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights, strict=False)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader, nice_test_loader = get_dataloader(args)
    # nice_deploy_loader = get_dataloader(args, deploy_mode=True)

    '''checkpoint path and tensorboard'''
    # checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))

    # create checkpoint folder to save model
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    # checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begin testing'''
    # best_acc = 0.0
    # best_tol = 1e4
    # best_dice = 0.0

    # test_sam(args, nice_deploy_loader, 0, net, writer)

    net.eval()
    tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, 0, net, writer)
    
    # logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice}.')

    # torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))

    writer.close()


if __name__ == '__main__':
    main()