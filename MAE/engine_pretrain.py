# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import datetime
import math
import sys
from typing import Iterable

import time

import torch
from tqdm import tqdm

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, rank_ctrl=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    use_mixed_precision = args.temp5 == 1
    norm_sums, last_loss_scale = {}, None  # feature norms go to log.txt only, not the console line
    epoch_start = time.time()
    bar = tqdm(data_loader, desc=header, dynamic_ncols=True, disable=not misc.is_main_process())
    raw_model = getattr(model, 'module', model)
    for data_iter_step, (samples, _) in enumerate(bar):
        # Task 2: ask the encoder to keep per-block tokens on measurement steps
        # rank_interval counts optimiser steps (one per accum_iter batches), so every epoch gets measured
        measure = rank_ctrl is not None and (data_iter_step + 1) % accum_iter == 0 and rank_ctrl.should_measure(
            (epoch * len(data_loader) + data_iter_step + 1) // accum_iter, epoch)
        raw_model.rank_collect = measure

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled = use_mixed_precision):
            loss, enc_feat_norm_list, dec_feat_norm_list = model(samples, mask_ratio=args.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        if use_mixed_precision:
            grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                    update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
                metric_logger.update(grad_norm=float(grad_norm))
                last_loss_scale = loss_scaler._scaler.get_scale()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        if measure:  # after backward/step, so the in-place decay update never touches a live graph
            rank_ctrl.update(raw_model, data_iter_step / len(data_loader) + epoch)
            raw_model.rank_collect = False

        for prefix, norms in (('enc', enc_feat_norm_list), ('dec', dec_feat_norm_list)):
            for fid, feat_norm in enumerate(norms):
                key = f'{prefix}_feat_norm_{fid}'
                norm_sums[key] = norm_sums.get(key, 0) + feat_norm.detach()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if data_iter_step % print_freq == 0:
            gn = metric_logger.meters.get('grad_norm')
            bar.set_postfix(loss=f'{metric_logger.meters["loss"].avg:.4f}', gnorm=f'{gn.value:.3f}' if gn else '-',
                            lr=f'{lr:.2e}', mem=f'{torch.cuda.max_memory_allocated() / 2**30:.1f}G')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    gn = metric_logger.meters.get('grad_norm')
    print('== epoch {} done | loss {:.4f} | gnorm {} | lr {:.2e} | {} | mem {:.1f}G =='.format(
        epoch, metric_logger.meters['loss'].global_avg, '{:.3f}'.format(gn.global_avg) if gn else '-', lr,
        str(datetime.timedelta(seconds=int(time.time() - epoch_start))), torch.cuda.max_memory_allocated() / 2**30))
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update({k: (v / len(data_loader)).item() for k, v in norm_sums.items()})
    if rank_ctrl is not None:
        stats.update(rank_ctrl.epoch_stats())
    if last_loss_scale is not None:
        stats['loss_scale'] = last_loss_scale
    return stats
