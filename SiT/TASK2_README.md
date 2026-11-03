# Task 2 on SiT: Rank-Targeted Residual Regulation

Same idea as `MAE/TASK2_README.md`, applied to the SiT diffusion transformer. The per-block residual decay `d`
(`x = d*x + f(x)`) is no longer a fixed schedule. A non-learned feedback controller measures the effective rank of each
block's output tokens and nudges that block's `d` toward a target-rank trajectory. Smaller `d` gives lower rank, so a block
whose rank is above target gets a smaller `d`, and a block whose rank is below target gets a larger `d`.

## What changed (and on which branch)

**Baseline changes (on `main`, also merged into `task2`)**
| File | Change |
|---|---|
| `models.py` | `flash_attn` is optional. If it is missing, `Attention` uses `F.scaled_dot_product_attention` (same math, max diff 1e-7). Needed to run on this RTX 3070. |
| `models.py` | `SiTBlock(decay_mode=...)`: `fixed` is the original behaviour. `learnable` stores `decay` as an `nn.Parameter` clamped to [0,1] (the learnable-decay baseline). |
| `train.py` | `--decay-mode {fixed,learnable}`; the mode is added to the experiment folder name. |

**Task 2 only (`task2` branch)**
| File | Change |
|---|---|
| `models.py` | `decay_mode='rank'` stores `decay` as a non-learned buffer. `SiT.rank_collect` makes `forward` keep the patch tokens of every block output for 8 images. A `blocks` property lists `enc_blocks + dec_blocks`. |
| `rank_control.py` (new) | `effective_rank()` (the paper's metric, `exp(entropy of normalised singular values)`) and `RankController`, which holds the target schedules and the feedback law. |
| `train.py` | `--decay-mode rank` plus the `--rank-*` options. The controller measures every `--rank-interval` optimiser steps, after `opt.step()`. The updated decays are copied into the EMA model, which otherwise only averages parameters. Each update is logged to `rank_log.jsonl`, with a per-epoch stability summary and the total controller time. |
| `eval_task2.py` (new) | KNN, linear probe, token-level and pooled effective rank per block, and the final decays, for any `--decay_mode`. |
| `scripts/run_task2.sh` (new) | One command that trains and then evaluates a given mode. |

No parameter is added or learned in `rank` mode, and no gradient flows through the controller.

## Controller
Every `rank-interval` optimiser steps (all ranks do this in sync, because the rank values are all-reduced):
```
obs_i   = erank(block_i tokens) / min(n_tokens, C)        # fraction of max rank, n = 8 images x 256 tokens (<= 2048)
ema_i   = 0.9*ema_i + 0.1*obs_i
err_i   = ema_i - target_i(depth, progress)
delta_i = 0 if |err_i| < deadband else clip(-gain*err_i, +-max_step)
d_i     = clip(d_i + delta_i, floor, 1)                    # both the attention and MLP decay of block i
```
Target schedules (`--rank-schedule`), from `--rank-target-start` (hi) to `--rank-target-end` (lo):
`const` (hi everywhere), `depth` (hi to lo from the first to the last block), `progress` (hi to lo over training, same for every block),
`depth_progress` (the depth slope grows with training progress).
The initial `d` is the original schedule (`linspace(1, min_sigma)`), so the controller starts from the paper's decay.

## How to run
```bash
cd SiT
bash scripts/run_task2.sh fixed                      # baseline 1: original decayed shortcuts
bash scripts/run_task2.sh learnable                  # baseline 2: learnable decay
bash scripts/run_task2.sh rank_depth                 # Task 2 (also: rank_const rank_progress rank_depth_progress)
# knobs: MODEL=SiT-S/2 EPOCHS=100 NGPU=1 BS=256 MICRO=32 DATA=../MAE/datasets/imagenet100 WANDB=1; extra args go to train.py, e.g.
bash scripts/run_task2.sh rank_depth --rank-gain 0.03 --rank-max-step 0.002 --rank-target-start 0.25 --rank-target-end 0.45
```
The script trains (`results_task2/<mode>/...`, controller trace in `rank_log.jsonl`), then writes `results_task2/<mode>.json`.
For the overhead comparison, compare "Train Steps/Sec" in the training logs and the "Rank controller overhead" line at the end.
The original paper command still works: `torchrun --nproc_per_node=4 train.py --model SiT-XL/2 --data-path <imagenet1k>/train --min-sigma 0.8`.

## Model and GPU memory (everything stays under 8 GB)
- **Model:** `SiT-S/2`, the smallest original SiT config (depth 12, hidden 384, 33M parameters). It is the default of `run_task2.sh` and `eval_task2.py`. SiT-XL/2 is 675M.
- **Training memory:** `train.py` accumulates gradients over micro-batches (`--micro-batch`, default 32). The effective batch stays `--global-batch-size` (256), so the setting is unchanged. The fp32 VAE encodes in chunks of `--vae-chunk` (8), and the memory cost is only speed. Measured on the 3070 (SiT-S/2, rank mode, effective batch 256): **peak 3.05 GiB at micro-batch 32**, 4.90 GiB at 64. Without accumulation, batch 128 and 256 ran out of memory on 8 GB. The training log prints "Peak GPU mem". If you see an OOM, lower `MICRO` (e.g. `MICRO=16`).
- **Eval memory:** `eval_task2.py` uses batch 16 with chunked VAE encoding. Peak **1.78 GiB** (stored in the result JSON as `peak_gpu_mem_gib`).
- **Speed on the 3070:** 0.16 steps/s at effective batch 256 (~41 img/s), so one ImageNet-100 epoch (126k images) takes ~50 min and 100 epochs ~85 h per run. Use fewer epochs (`EPOCHS=...`, the same for every mode) on this GPU, or run on the A6000. Eval over the full train set takes ~45 to 60 min.

## Downloaded models
`python -c "from download import find_model; find_model('SiT-XL-2-256x256.pt')"` puts the original SiT-XL/2 weights in
`pretrained_models/` (2.7 GB). The VAEs `stabilityai/sd-vae-ft-ema` and `-mse` are cached in `~/.cache/huggingface`.
The original SiT-XL/2 has no U-Net skips and no decay, so it is only a reference. It does not load into this architecture.
Needed in the env you run it from: `diffusers`, `torchdiffeq`, `timm`, `tqdm` (`pip install diffusers torchdiffeq`). `flash_attn` is optional.

## Tested here (RTX 3070, 8 GB)
- `fixed` and `learnable` models: forward/backward OK; learnable decay receives gradients.
- `rank` mode: ~2k steps of SiT-S/2 (batch 8, ImageNet-100): the controller updates every block, `rank_log.jsonl` is written, checkpoints are saved, and EMA decays are synced.
- `eval_task2.py` on that checkpoint: KNN, probe, erank and decays all produced (accuracies are near chance because training was tiny).
- Accumulation run: effective batch 256 with micro-batch 32 for 30 steps, peak 3.05 GiB.
- **Not run:** full trainings, so no result numbers exist yet.

## Suggested order
1. `conda activate SiT`, check the packages above.
2. `EPOCHS=<N> bash scripts/run_task2.sh fixed`, then read `results_task2/fixed.json` (`erank_token_frac_per_block`) to calibrate the targets.
3. `learnable`, then the four `rank_*` modes with the calibrated `--rank-target-start/--rank-target-end` (same `EPOCHS`, same seed).
4. Compare the six JSONs (KNN, probe, erank) and the `Train Steps/Sec` lines; plot `rank_log.jsonl` for the decay stability.

## Caveats and to-do
1. **Calibrate the targets.** The default 0.20 to 0.40 comes from MAE. In the smoke test the observed SiT erank fraction was ~0.26 to 0.29, below the targets, so every `d` was pushed to the cap of 1.0 (no decay). First run `fixed`, read its per-block erank from `results_task2/fixed.json`, and set `--rank-target-start/--rank-target-end` around those values (the MAE README did the same with `fixed_decay07_calib.json`).
2. Effective rank depends on the noise level. Training measures at random `t`; `eval_task2.py` measures at `--t 0.8`. Compare models only at the same `t`.
3. Probe/KNN use mean-pooled tokens of intermediate blocks with the null class label, at one noise level. This is our protocol and not necessarily the paper's exact one.
4. Fill in the results table (fixed / learnable / rank_* by KNN, probe, erank, steps/s) after the runs.
