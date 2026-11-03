# Task 2 — Rank-Targeted Residual Regulation

Branch: `task2`. The fixed decay schedule of the original method is replaced by a **non-learned feedback
controller** that sets the residual decay of every encoder block from the *measured effective rank* of
that block's features. (Task 1, frequency-selective decay, is independent and lives on another branch.)

## Idea
The paper observes: stronger decay (smaller `d` in `x <- d*x + f(x)`) -> lower effective rank -> better features.
So the decay is turned into the actuator of a control loop whose sensor is effective rank:

```
every rank_interval steps (after warm-up):
  erank_i  = exp( H(sigma/sum sigma) )          # effective rank of block i's patch tokens (same metric as the paper / RankMe)
  obs_i    = erank_i / min(n_tokens, C)          # fraction of the maximum rank, EMA-smoothed
  err_i    = obs_i - target_i(depth, progress)
  delta_i  = 0                         if |err_i| < deadband
           = clip(-gain * err_i, +-max_step)     # rank too high -> decay more (smaller d); too low -> relax
  d_i     <- clip(d_i + delta_i, floor, 1)       # applied to both the attention and MLP decay of block i
```
It is an integral controller: no parameters are learned, no gradient flows through it, ranks are all-reduced
over GPUs so every process applies the identical update. The update happens *after* the optimiser step, so it
never touches a live autograd graph. Initial decays = the original schedule (`--min_decay`).

## Target-rank schedules (`--rank_schedule`)
Targets are fractions of the maximum rank (`--rank_target_start` -> `--rank_target_end`):

| schedule | target |
|---|---|
| `const` | `start` for every block, always |
| `depth` (default) | linear `start -> end` from first to last block (fixed over training) |
| `progress` | same for all blocks, moves `start -> end` over the training run |
| `depth_progress` | depth profile that goes from flat (`start`) to the full `start -> end` slope during training |

**Calibration.** I measured the already-trained fixed-decay model (`out_decay07`, epoch 100): token-level
effective-rank fraction per block = `0.23 0.28 0.33 0.38 0.39 0.44 0.47 0.52 0.53 0.56 0.52 0.51`
(`results/fixed_decay07_calib.json`). Rank rises with depth, so the defaults (`0.20 -> 0.40`) ask for roughly
20-25 % lower rank than the fixed schedule yields. Try other targets (e.g. `0.15 -> 0.30`, or `const 0.25`) as the schedule study.
Note training measures masked inputs (25 % visible tokens), evaluation uses full images, so absolute values differ a bit.

## Files changed / added
| file | change |
|---|---|
| `models_mae.py` | `Block` got `decay_mode`: `fixed` (original, unchanged behaviour), `learnable` (decay = `nn.Parameter`, clamped to [0,1] -> learnable-decay baseline), `rank` (decay = non-learned **buffer**, so it is saved in checkpoints and reloaded for eval). `MaskedAutoencoderViT` takes `decay_mode`, and with `rank_collect=True` keeps each block's detached patch tokens. |
| `util/rank_control.py` (new) | `effective_rank()` and `RankController` (targets, EMA, dead band, rate limit, per-epoch stability stats). |
| `engine_pretrain.py` | On measurement steps turns on feature collection, then calls `rank_ctrl.update` after the optimiser step; adds controller stats to the epoch log. |
| `main_pretrain.py` | New args `--decay_mode --rank_schedule --rank_target_start/end --rank_gain --rank_max_step --rank_deadband --rank_ema --rank_decay_floor --rank_interval --rank_warmup_epochs --rank_max_tokens`; builds the controller; saves/restores its state (`rank_ctrl.pt`); output dir gets a `_rank_<schedule>` / `_learnable` suffix. |
| `main_eval.py` | `--decay_mode` so linear probing rebuilds the model with the stored decays (errors out if you forget it). |
| `eval_task2.py` (new) | Single-GPU evaluation: **KNN accuracy** (k=20, cosine, T=0.07) and **effective rank per block** (token-level fraction + pooled image-level) and the final decays -> JSON. |
| `scripts/run_task2.sh` (new) | Runs training then both evaluations. |

Everything else (data, augmentation, optimiser, lr, epochs, masking) is the unchanged MAE setting.

## How to run
```bash
cd MAE
bash scripts/run_task2.sh rank_depth            # main Task-2 method
bash scripts/run_task2.sh rank_const            # other target schedules
bash scripts/run_task2.sh rank_progress
bash scripts/run_task2.sh rank_depth_progress
bash scripts/run_task2.sh learnable             # learnable-decay baseline
# override anything, e.g.:
bash scripts/run_task2.sh rank_depth --rank_target_start 0.15 --rank_target_end 0.3 --rank_gain 0.2 --wandb
# the original fixed-decay baseline is your existing out_decay07 run; evaluate it the same way:
python eval_task2.py --ckpt out_decay07/*/checkpoint-last.pth --decay_mode fixed --out results/fixed.json
bash scripts/eval_ckpt.sh out_decay07/*/checkpoint-last.pth mae_vit_base_patch16 ./datasets/imagenet100
```
Main entry points: **`scripts/run_task2.sh`** (everything) -> `main_pretrain.py` (training) -> `eval_task2.py` + `main_eval.py` (evaluation).

## What to report
* **Linear probe**: printed by `main_eval.py` (`accuracy`, per layer) in `output_dir/eval_*/console.log`.
* **KNN / effective rank**: `results/<tag>.json` (`knn_acc`, `erank_pooled`, `erank_token_frac_per_block`, `decay_per_block`).
* **Stability of the scaling factors**: `out_task2_*/*/log.txt` has per epoch `train_rank_decay_<i>` (decay trajectory),
  `train_rank_decay_std_<i>` (jitter inside an epoch), `train_rank_obs_<i>` vs `train_rank_target_<i>`,
  `train_rank_abs_err`, `train_rank_abs_delta` (all go to wandb with `--wandb`).
* **Training overhead**: `train_rank_ctrl_time_s` (controller time per epoch) and `epoch_time_s` / `img_per_s` vs the baseline log.

## Results so far (ImageNet-100, ViT-B/16, 100 epochs, single seed)
KNN = k=20 cosine over the full 126k train set, last block, mean-pooled. Effective rank = image-level pooled features on the 5000 val images (max 768).
Source: `results/<tag>.json`.

| run | KNN acc | eff. rank (pooled) | token-level rank, first -> last block | final decays (mean of attn/mlp), first -> last block | status |
|---|---|---|---|---|---|
| fixed decay (original, `out_decay07`) | **48.0 %** | 281 | 0.23 -> 0.51 | 1.0 -> 0.7 (fixed schedule) | done |
| learnable decay | 44.2 % | 298 | 0.24 -> 0.59 | 1.0 ... 0.71 (learned; early blocks drifted to 1.0) | done |
| **rank-targeted, `rank_depth`** (buggy controller, see below) | **48.3 %** | **244** | 0.18 -> 0.38 | alternating 1.0 / 0.5 (saturated) | done |
| rank_const / rank_progress / rank_depth_progress | - | - | - | - | **not run yet** |
| linear probe (all runs) | - | - | - | - | **not run yet** (see below) |

Training overhead (`rank_depth`): about 470 s/epoch vs 465 s for the baseline; the controller itself costs ~12 s on epochs where it measures (~2.5 %).

### Honest reading
* The method does what it is designed to do: the rank was pushed down (pooled 281 -> 244, token rank 0.51 -> 0.38 in the last block) and the targets were roughly tracked.
* KNN is essentially **tied** with the fixed schedule (48.3 vs 48.0). A 0.3-point gap on one seed is within noise, so this is **not yet evidence of an improvement**. It does clearly beat the learnable-decay baseline (44.2 %), which left the rank high (298).
* The `rank_depth` run used a controller with two known problems (below), so treat it as a first, pessimistic data point.

### Known issues
1. **Decay saturation** - the final decays flip between 1.0 and the 0.5 floor instead of settling, i.e. the gain is too aggressive / the target too low for some blocks. Next try: `--rank_gain 0.03 --rank_max_step 0.002 --rank_decay_floor 0.6` and/or less aggressive targets (e.g. `--rank_target_start 0.22 --rank_target_end 0.45`).
2. **Measurement skipped on alternate epochs** (batches/epoch not a multiple of `accum_iter`). *Fixed* in the latest commit (`--rank_interval` now counts optimiser steps, default 25). The finished `rank_depth` run predates the fix, so it should be re-run.

## What still needs to be run
Each run is ~13 h on an 8 GB GPU and needs ~6.5 GB, so one per GPU. Run from `MAE/` with `scripts/run_task2.sh MODE [extra args]` (`CUDA_VISIBLE_DEVICES=N` to pick a GPU; every mode writes to its own `out_task2_<tag>/`). Training, KNN/rank eval and linear probe all run from that one script.

| # | command | purpose |
|---|---|---|
| 1 | `bash scripts/run_task2.sh rank_depth --rank_gain 0.03 --rank_max_step 0.002 --rank_decay_floor 0.6` | **re-run main method** with the fixed trigger and a gentler controller |
| 2 | `bash scripts/run_task2.sh rank_const` | target schedule: constant |
| 3 | `bash scripts/run_task2.sh rank_progress` | target schedule: annealed over training |
| 4 | `bash scripts/run_task2.sh rank_depth_progress` | target schedule: depth + progress |
| 5 | linear probe on the finished checkpoints (below) | missing metric for fixed / learnable / rank_depth |

Linear probes for already-trained checkpoints (needs the conda env first: `export PATH=/home/dse_413/.conda/envs/mae/bin:$PATH`):
```bash
bash scripts/eval_ckpt.sh out_decay07/*/checkpoint-last.pth                     mae_vit_base_patch16 ./datasets/imagenet100 --min_decay 0.7
bash scripts/eval_ckpt.sh out_task2_learnable/*/checkpoint-last.pth             mae_vit_base_patch16 ./datasets/imagenet100 --min_decay 0.7 --decay_mode learnable
bash scripts/eval_ckpt.sh out_task2_rank_depth/*/checkpoint-last.pth            mae_vit_base_patch16 ./datasets/imagenet100 --min_decay 0.7 --decay_mode rank
```
Results land in `output_dir/eval_*/console.log` (`idx:11, accuracy:...` = last block). The sequential queue `scripts/run_task2_all.sh` runs all four trainings back to back on one GPU (wandb removed because the network here breaks its TLS).
After the runs: compare `results/*.json` (KNN, rank), the linear-probe logs, and the decay/rank trajectories in `out_task2_*/*/log.txt` (`train_rank_*` keys; stability = `train_rank_decay_std_<i>`, `train_rank_abs_delta`). Ideally add 2-3 seeds (`--seed`) for the final table, since differences are small.

## Tested
Controller unit test (4-block model, no autograd errors), a 3-epoch end-to-end smoke run (training, controller, resume, validation), `eval_task2.py` on smoke and trained checkpoints, and full 100-epoch runs of `rank_depth` and `learnable`.
