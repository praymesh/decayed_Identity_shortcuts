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
import hashlib
import random
import signal
from collections import Counter
import numpy as np
import os
import time
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

from engine_pretrain import train_one_epoch, evaluate
from util.run_log import RunLog

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training')
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
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--memory_budget_gib', default=7.0, type=float)
    parser.add_argument('--synthetic_samples', default=0, type=int,
                        help='Synthetic smoke check only; never an ImageNet result')
    parser.add_argument('--val_path', default='', type=str)
    parser.add_argument('--class_list', default='', type=str,
                        help='Agreed list of 100 synset IDs, one per line')
    parser.add_argument('--dataset_id', default='', type=str,
                        help='Dataset source and version/revision')
    parser.add_argument('--val_seed', default=12345, type=int)
    parser.add_argument('--val_samples', default=0, type=int,
                        help='Deterministic validation subset; zero uses all images')
    parser.add_argument('--train_samples_per_class', default=0, type=int,
                        help='Exploratory balanced subset; zero uses all training images')
    # mixed precision
    parser.add_argument('--temp5', default=1.0, type=float,
                        help='parameter_5')
    parser.add_argument('--eval_epochs_interval', default=5, type=int,
                        help='MAE validation interval; also validates the final epoch')
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

    parser.add_argument('--output_dir', default=str(Path(__file__).resolve().parents[1] / 'runs'),
                        help='Parent folder for unique runs')
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

def run_training(args, log_writer):
    args.output_dir = str(log_writer.root / 'checkpoints')
    args.log_dir = str(log_writer.root / 'logs')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if device.type != 'cuda' and args.temp5 == 1:
        raise ValueError('Use --temp5 0 for CPU checks')
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        fraction = args.memory_budget_gib * 2**30 / total_memory
        if not 0 < fraction <= 1:
            raise ValueError('Memory budget must fit on the actual GPU')
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        torch.cuda.reset_peak_memory_stats(device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3), transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if args.synthetic_samples:
        dataset_train = datasets.FakeData(args.synthetic_samples, (3, args.input_size, args.input_size),
                                          num_classes=100, transform=transform_train, random_offset=0)
        dataset_val = datasets.FakeData(8, (3, args.input_size, args.input_size),
                                       num_classes=100, transform=transform_val, random_offset=1000000)
        dataset_record = dict(kind='SYNTHETIC_SMOKE_ONLY', train_images=len(dataset_train),
                              validation_images=len(dataset_val), subset=True)
    else:
        if not args.class_list or not args.dataset_id:
            raise ValueError('Real runs require --class_list and --dataset_id; no implicit class subset')
        classes = sorted(Path(args.class_list).read_text().split())
        if len(classes) != 100 or len(set(classes)) != 100:
            raise ValueError('Class list must contain exactly 100 unique synset IDs')
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
        val_path = args.val_path or str(Path(args.data_path).resolve().parent / 'val')
        dataset_val = datasets.ImageFolder(val_path, transform=transform_val)
        if dataset_train.classes != classes or dataset_val.class_to_idx != dataset_train.class_to_idx:
            raise ValueError('Train/validation class sets and label mappings must match the agreed list')
        dataset_record = dict(kind='ImageNet-100', version=args.dataset_id,
                              class_to_idx=dataset_train.class_to_idx,
                              train_counts=dict(Counter(dataset_train.targets)),
                              validation_counts=dict(Counter(dataset_val.targets)),
                              train_images=len(dataset_train), validation_images=len(dataset_val),
                              class_list_sha256=hashlib.sha256(Path(args.class_list).read_bytes()).hexdigest(),
                              subset=bool(args.val_samples))
        if args.train_samples_per_class:
            generator = np.random.default_rng(args.seed)
            targets = np.asarray(dataset_train.targets)
            indices = []
            for label in range(100):
                candidates = np.flatnonzero(targets == label)
                if len(candidates) < args.train_samples_per_class:
                    raise ValueError('Requested more training images than a class contains')
                indices.extend(generator.choice(candidates, args.train_samples_per_class, replace=False).tolist())
            dataset_train = torch.utils.data.Subset(dataset_train, indices)
            dataset_record.update(train_indices=indices, subset=True, used_train_images=len(indices))
    if args.val_samples:
        # Evenly spaced fixed indices avoid selecting only the first few class folders.
        indices = np.linspace(0, len(dataset_val) - 1, min(args.val_samples, len(dataset_val)), dtype=int).tolist()
        dataset_val = torch.utils.data.Subset(dataset_val, indices)
        dataset_record['validation_indices'] = indices
    args.dataset_fingerprint = hashlib.sha256(str(dataset_record).encode()).hexdigest()
    log_writer.save_metadata(dataset=dataset_record)
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

    loader_generator = torch.Generator()
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=False,
        generator=loader_generator,
    )
    if not len(data_loader_train):
        raise ValueError('Training data is smaller than one physical batch')
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, shuffle=False, generator=torch.Generator().manual_seed(args.val_seed))

    model = models_mae.__dict__[args.model](img_size=args.input_size, norm_pix_loss=args.norm_pix_loss,
                                          min_decay=args.min_decay, beta=args.beta)

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

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    log_writer.save_metadata(settings=vars(args).copy(), method=model_without_ddp.method_settings(),
                             method_name=('standard' if args.min_decay == 1 else
                                          'fixed_decay' if args.beta == 0 else 'frequency_selective'),
                             effective_batch_size=eff_batch_size,
                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f"Start training at {args.start_epoch} for total {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # misc.eval_model(args, epoch, model)
        data_loader_train.sampler.set_epoch(epoch)
        loader_generator.manual_seed(args.seed + epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        log_writer.record('train_epoch', epoch=epoch, optimizer_updates=args.optimizer_updates,
                          mae_train_loss=train_stats['loss'], lr=train_stats['lr'])
        if ((args.eval_epochs_interval > 0 and (epoch + 1) % args.eval_epochs_interval == 0)
                or epoch + 1 == args.epochs):
            evaluate(model, data_loader_val, device, args, epoch, log_writer)
        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_last = True)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def main(args):
    misc.init_distributed_mode(args)
    if args.distributed:
        raise ValueError('The logged training runner currently supports one GPU per run')
    if args.accum_iter < 1 or args.batch_size < 1 or not 0 < args.mask_ratio < 1:
        raise ValueError('Invalid accumulation, batch size or mask ratio')
    args.optimizer_updates = 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    log_writer = RunLog(args.output_dir, args)
    print(f'Run folder: {log_writer.root}')
    try:
        run_training(args, log_writer)
    except BaseException as error:
        log_writer.close('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', str(error))
        raise
    else:
        log_writer.close('completed')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
