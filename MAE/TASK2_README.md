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

## Tested
Controller unit test (4-block model, no autograd errors, decays move monotonically), a 3-epoch end-to-end smoke
run (training, controller, resume file, validation) and `eval_task2.py` on both a smoke checkpoint and the trained
baseline. A full 100-epoch rank-targeted run has **not** been trained yet.
