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
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm


#assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch
from util.plot_log import plot_log

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--min_decay', default=1.0, type=float,
                        help='minimum decay rate for the skip connection')
    # Task 2: residual decay mode / Rank-Targeted Residual Regulation
    parser.add_argument('--decay_mode', default='fixed', choices=['fixed', 'learnable', 'rank'],
                        help="fixed: original schedule; learnable: learnable-decay baseline; rank: Task 2 rank-targeted feedback")
    parser.add_argument('--rank_schedule', default='depth', choices=['const', 'depth', 'progress', 'depth_progress'],
                        help='target effective-rank trajectory (fraction of max rank) over depth / training progress')
    parser.add_argument('--rank_target_start', default=0.20, type=float, help='target erank fraction (of max rank) at first block / at start of training')
    parser.add_argument('--rank_target_end', default=0.40, type=float, help='target erank fraction at last block / at end of training')
    parser.add_argument('--rank_gain', default=0.1, type=float, help='integral gain of the decay controller')
    parser.add_argument('--rank_max_step', default=0.005, type=float, help='max |change| of a decay per update')
    parser.add_argument('--rank_deadband', default=0.01, type=float, help='no update if |erank - target| below this')
    parser.add_argument('--rank_ema', default=0.9, type=float, help='EMA factor for the observed erank')
    parser.add_argument('--rank_decay_floor', default=0.5, type=float, help='lowest decay value the controller may set')
    parser.add_argument('--rank_interval', default=25, type=int, help='measure effective rank every N optimiser steps')
    parser.add_argument('--rank_warmup_epochs', default=2, type=int, help='no control during the first N epochs')
    parser.add_argument('--rank_max_tokens', default=2048, type=int, help='tokens subsampled for the SVD')
    # mixed precision
    parser.add_argument('--temp5', default=1.0, type=float,
                        help='parameter_5')
    parser.add_argument('--eval_epochs_interval', default=20, type=int,
                        help='run validation every N epochs')
    parser.add_argument('--val_path', default='', type=str,
                        help='validation folder (default: "val" sibling of --data_path, e.g. imagenet/val)')
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
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
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

    # wandb (optional)
    parser.add_argument('--wandb', action='store_true', help='log per-epoch stats to Weights & Biases')
    parser.add_argument('--wandb_project', default='mae-decayed-identity-shortcuts', type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='run name (default: basename of output_dir)')

    parser.add_argument('--debug', action='store_true', help='flag for running debugging experiments')

    return parser

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

@torch.no_grad()
def evaluate(model, data_loader, device, args):
    """Mean MAE reconstruction loss on the held-out set (fixed masking seed for comparability)."""
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    n_sum = torch.zeros(1, device=device)
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(args.seed)
        for samples, _ in tqdm(data_loader, desc='Eval', dynamic_ncols=True, disable=not misc.is_main_process()):
            samples = samples.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.temp5 == 1):
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            loss_sum += loss.detach().float() * samples.shape[0]
            n_sum += samples.shape[0]
    if args.distributed:
        torch.distributed.all_reduce(loss_sum)
        torch.distributed.all_reduce(n_sum)
    model.train(True)
    return {'val_loss': (loss_sum / n_sum).item()}


def main(args):
    #handle_flash_attn()
    args.output_dir = '''{}/birdge_dataset_lr_{}_batch_size_{}_min_decay_{}'''.replace('\n',' ').replace(' ','').format(
                        args.log_dir, args.blr, args.batch_size, args.min_decay)
    if args.decay_mode != 'fixed':  # keep Task 2 / learnable runs apart from the original directory naming
        args.output_dir += '_' + args.decay_mode + ('_' + args.rank_schedule if args.decay_mode == 'rank' else '')
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

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers = True
    )

    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_path = args.val_path or os.path.join(os.path.dirname(os.path.normpath(args.data_path)), 'val')
    dataset_val = datasets.ImageFolder(val_path, transform=transform_val)
    print('val set: %s (%d images)' % (val_path, len(dataset_val)))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=False),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, min_decay = args.min_decay, decay_mode = args.decay_mode)

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
    if misc.is_main_process():
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump({**vars(args), 'eff_batch_size': eff_batch_size}, f, indent=2, default=str)

    use_wandb = args.wandb and misc.is_main_process()
    if use_wandb:
        import wandb
        # persist the run id next to the checkpoints so --resume continues the same wandb run
        id_file = os.path.join(args.output_dir, 'wandb_id.txt')
        if os.path.exists(id_file):
            run_id = open(id_file).read().strip()
        else:
            run_id = wandb.util.generate_id()
            with open(id_file, 'w') as f:
                f.write(run_id)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or os.path.basename(args.output_dir),
                   id=run_id, resume='allow', dir=args.output_dir, config={**vars(args), 'eff_batch_size': eff_batch_size})
        wandb.define_metric('epoch')
        wandb.define_metric('*', step_metric='epoch')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    rank_ctrl = None
    if args.decay_mode == 'rank':
        from util.rank_control import RankController
        rank_ctrl = RankController(model_without_ddp, args)
        ctrl_file = os.path.join(args.output_dir, 'rank_ctrl.pt')
        if args.resume and os.path.exists(ctrl_file):
            rank_ctrl.load_state_dict(torch.load(ctrl_file, map_location='cpu'))

    print(f"Start training at {args.start_epoch} for total {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # misc.eval_model(args, epoch, model)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats()
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args, rank_ctrl=rank_ctrl
        )
        epoch_time = time.time() - epoch_start
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'lr_end': optimizer.param_groups[0]['lr'],
                     'epoch_time_s': epoch_time,
                     'img_per_s': len(data_loader_train) * args.batch_size * misc.get_world_size() / epoch_time,
                     'max_mem_mb': torch.cuda.max_memory_allocated() / 2**20,
                     'elapsed_s': time.time() - start_time}
        if (epoch + 1) % args.eval_epochs_interval == 0 or epoch + 1 == args.epochs:
            val_stats = evaluate(model, data_loader_val, device, args)
            log_stats.update(val_stats)
            print('Epoch %d: %s' % (epoch, val_stats))
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
            plot_log([args.output_dir], os.path.join(args.output_dir, 'training_curves.png'))
            if use_wandb:
                wandb.log(log_stats)
        if rank_ctrl is not None and misc.is_main_process():
            torch.save(rank_ctrl.state_dict(), os.path.join(args.output_dir, 'rank_ctrl.pt'))
        if args.output_dir and ((epoch + 1) % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if args.output_dir:  # last checkpoint every epoch so a crash loses at most one epoch
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_last = True)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
