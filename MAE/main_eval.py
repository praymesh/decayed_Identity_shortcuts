# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets


#assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--probe_batch_size', default=256, type=int)
    parser.add_argument('--probe_epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--temp1', default=1.0, type=float,
                        help='parameter_1')
    parser.add_argument('--temp2', default=1.0, type=float,
                        help='parameter_2')
    parser.add_argument('--temp3', default=1.0, type=float,
                        help='parameter_3')
    parser.add_argument('--temp4', default=1.0, type=float,
                        help='parameter_4')
    parser.add_argument('--temp5', default=0, type=float,
                        help='parameter_5')
    parser.add_argument('--min_decay', default=1.0, type=float,
                        help='minimum residual decay factor of the model being evaluated (1.0 = plain residual)')
    parser.add_argument('--decay_mode', default='fixed', choices=['fixed', 'learnable', 'rank'],
                        help='must match the pretraining run (rank/learnable store their decays in the checkpoint)')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/dse_413/ADL_Project/decayed_Identity_shortcuts/MAE/datasets/imagenet100', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/home/dse_413/ADL_Project/decayed_Identity_shortcuts/MAE/out_decay07/birdge_dataset_lr_0.00015_batch_size_64_min_decay_0.7/checkpoint-last.pth',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--eval_only', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    if not args.resume:
        raise ValueError('--resume must point to a pretrained checkpoint')
    if min(args.batch_size, args.probe_batch_size, args.probe_epochs) < 1 or args.num_workers < 0:
        raise ValueError('Batch sizes and probe epochs must be positive; workers must be nonnegative')
    misc.init_distributed_mode(args)
    if misc.is_main_process():
        if not os.path.exists(args.output_dir):
            os.system('mkdir -p {}'.format(args.output_dir))
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # define the model
    print(args.model)
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, min_decay=args.min_decay, decay_mode=args.decay_mode)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    encoder_missing = [key for key in msg.missing_keys
                       if key.startswith(('patch_embed.', 'blocks.', 'pos_embed', 'cls_token'))]
    if encoder_missing:
        raise ValueError(f'Checkpoint is missing encoder weights: {encoder_missing}')
    print(f'Loaded checkpoint; missing keys: {msg.missing_keys}; unexpected keys: {msg.unexpected_keys}')
    del checkpoint
    if args.decay_mode == 'fixed' and any('.decay' in k for k in msg.unexpected_keys):
        raise ValueError('Checkpoint stores per-block decays; pass --decay_mode rank (or learnable)')
    model_without_ddp.requires_grad_(False)
    misc.eval_model(args, args.start_epoch, model)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
