"""Task 2: Rank-Targeted Residual Regulation.

A non-learned feedback controller that rewrites the per-block residual decay
factors (the `decay` buffers of `models.SiTBlock`, decay_mode='rank') from the
measured effective rank of the intermediate encoder features.

    stronger decay (smaller d)  ->  lower effective rank   (observation in the paper)

so every `rank_interval` optimiser steps we compare the observed effective rank of
each block output with a target and nudge that block's decay (integral controller):

    err_i   = EMA(erank_i / max_rank) - target_i(depth, progress)
    delta_i = clip(-gain * err_i, +-max_step)           (0 inside the dead band)
    d_i    <- clip(d_i + delta_i, d_floor, 1.0)         (both attn and mlp decay of block i)

No parameter is learned and no gradient flows through the controller.
"""
import math
import time

import torch
import torch.distributed as dist


@torch.no_grad()
def effective_rank(x, max_tokens=2048):
    """Effective rank of a feature matrix x: (n, C)  [Roy & Vetterli 2007, as used by RankMe].

    erank = exp(H(p)),  p_k = sigma_k / sum_j sigma_j,  H = Shannon entropy.
    Returns (erank, max_possible_rank = min(n, C)).
    """
    x = x.float()
    if x.shape[0] > max_tokens:
        x = x[torch.randperm(x.shape[0], device=x.device)[:max_tokens]]
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return torch.exp(ent), float(min(x.shape))


class RankController:
    SCHEDULES = ('const', 'depth', 'progress', 'depth_progress')

    def __init__(self, model, args):
        """model: the un-wrapped SiT built with decay_mode='rank' (blocks = enc_blocks + dec_blocks)."""
        self.blocks = list(model.blocks)
        assert all(b.decay_mode == 'rank' for b in self.blocks), "build the model with decay_mode='rank'"
        self.depth = len(self.blocks)
        self.schedule = args.rank_schedule
        assert self.schedule in self.SCHEDULES
        self.hi, self.lo = args.rank_target_start, args.rank_target_end
        self.gain, self.max_step = args.rank_gain, args.rank_max_step
        self.deadband, self.beta = args.rank_deadband, args.rank_ema
        self.d_floor = args.rank_decay_floor
        self.interval = args.rank_interval
        self.warmup_epochs = args.rank_warmup_epochs
        self.total_epochs = args.epochs  # progress = epoch / epochs (SiT is trained by epochs too)
        self.max_tokens = args.rank_max_tokens
        self.ema = None
        self.last = {}                 # latest per-block observed / target / decay
        self._reset_epoch()
        self.total_time = 0.0

    # ---------------------------------------------------------------- targets
    def target(self, i, progress):
        """Target effective rank of block i as a fraction of its maximum possible rank."""
        frac_depth = i / max(self.depth - 1, 1)
        if self.schedule == 'const':
            return self.hi
        if self.schedule == 'depth':          # fixed profile start -> end across depth
            return self.hi + (self.lo - self.hi) * frac_depth
        if self.schedule == 'progress':       # same target for all blocks, start -> end over training
            return self.hi + (self.lo - self.hi) * progress
        # depth_progress: the depth profile gets steeper as training advances
        lo_t = self.hi + (self.lo - self.hi) * progress
        return self.hi + (lo_t - self.hi) * frac_depth

    # --------------------------------------------------------------- measuring
    def should_measure(self, step, epoch):  # step = optimiser-step index
        return epoch >= self.warmup_epochs and step % self.interval == 0

    @torch.no_grad()
    def measure(self, model):
        """erank / max_rank for every block, averaged over ranks. Needs model.rank_feats."""
        obs = []
        for f in model.rank_feats:
            r, m = effective_rank(f.reshape(-1, f.shape[-1]), self.max_tokens)
            obs.append(r / m)
        obs = torch.stack(obs)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(obs)
            obs /= dist.get_world_size()
        model.rank_feats = []
        return obs.cpu()

    # ----------------------------------------------------------------- control
    @torch.no_grad()
    def update(self, model, epoch_float):
        t0 = time.time()
        obs = self.measure(model)
        self.ema = obs if self.ema is None else self.beta * self.ema + (1 - self.beta) * obs
        progress = min(epoch_float / max(self.total_epochs, 1), 1.0)
        for i, blk in enumerate(self.blocks):
            tgt = self.target(i, progress)
            err = float(self.ema[i]) - tgt
            delta = 0.0 if abs(err) < self.deadband else max(-self.max_step, min(self.max_step, -self.gain * err))
            blk.decay.copy_((blk.decay + delta).clamp(self.d_floor, 1.0))
            d = blk.decay.mean().item()
            self.last[i] = dict(obs=float(obs[i]), ema=float(self.ema[i]), target=tgt, decay=d, err=err, delta=delta)
            self._ep_decay[i].append(d)
            self._ep_err[i].append(abs(err))
            self._ep_delta[i].append(abs(delta))
        self.total_time += time.time() - t0
        self._ep_time += time.time() - t0

    # ----------------------------------------------------------------- logging
    def _reset_epoch(self):
        self._ep_decay = [[] for _ in range(self.depth)]
        self._ep_err = [[] for _ in range(self.depth)]
        self._ep_delta = [[] for _ in range(self.depth)]
        self._ep_time = 0.0

    def epoch_stats(self):
        """Per-epoch log entries (decay trajectory + stability: std/|delta| of the decay within the epoch)."""
        out = {}
        for i in range(self.depth):
            d = torch.tensor(self._ep_decay[i]) if self._ep_decay[i] else torch.tensor([self.blocks[i].decay.mean().item()])
            out[f'rank_decay_{i}'] = d[-1].item()
            out[f'rank_decay_std_{i}'] = d.std().item() if len(d) > 1 else 0.0
            if i in self.last:
                out[f'rank_obs_{i}'] = self.last[i]['ema']
                out[f'rank_target_{i}'] = self.last[i]['target']
        errs = [e for l in self._ep_err for e in l]
        deltas = [e for l in self._ep_delta for e in l]
        out['rank_abs_err'] = sum(errs) / len(errs) if errs else 0.0
        out['rank_abs_delta'] = sum(deltas) / len(deltas) if deltas else 0.0
        out['rank_ctrl_time_s'] = self._ep_time
        self._reset_epoch()
        return out

    def state_dict(self):
        return {'ema': self.ema, 'total_time': self.total_time}

    def load_state_dict(self, sd):
        self.ema, self.total_time = sd['ema'], sd.get('total_time', 0.0)
