import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

import cfg
import func_2d.function as function
from conf import settings
from func_2d import utils as fn2dutils
from func_2d.inference import MedicalSam2ImagePredictor

# from models.discriminatorlayer import discriminator
from func_2d.dataset import REFUGE

import matplotlib.pyplot as plt
from pathlib import Path


default_args = argparse.Namespace(
    model_id="sam2",
    encoder="vit_b",
    exp_name="REFUGE_MedSAM2",
    vis=1,
    prompt="bbox",
    prompt_freq=2,
    val_freq=1,
    gpu=True,
    gpu_device=0,
    image_size=1024,
    out_size=1024,
    distributed="none",
    dataset="REFUGE",
    data_path="data\\REFUGE",
    sam_ckpt="checkpoints\\sam2_hiera_tiny.pt",
    sam_config="sam2_hiera_t",
    video_length=2,
    b=4,
    lr=1e-4,
    weights="0",
    multimask_output=1,
    memory_bank_size=16,
)

if __name__ == '__main__':
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    GPUdevice = torch.device("cuda", default_args.gpu_device)

    transform_test = transforms.Compose([
        transforms.Resize((default_args.image_size, default_args.image_size)),
        transforms.ToTensor(),
    ])

    refuge_test_dataset = REFUGE(
        default_args, default_args.data_path, transform=transform_test, mode="Test"
    )

    nice_test_loader = DataLoader(
        refuge_test_dataset, batch_size=default_args.b, shuffle=False, num_workers=4, pin_memory=True
    )

    iterator = iter(nice_test_loader)

    net = fn2dutils.get_network(default_args, default_args.model_id, gpu_device=GPUdevice,
                                distribution=default_args.distributed)
    predictor = MedicalSam2ImagePredictor(net, default_args)
    data_loaded = next(iterator)
    predictor.set_image_batch(data_loaded["image"])
    pred_mask, pred, high_res_multimasks = predictor.predict_batch()