## Moving to the A6000 machine (magic-wormhole) and run order
Batch size is unchanged (`--batch_size 64 --accum_iter 4`, effective 256). Do **not** rerun everything: only the linear probes need the old checkpoints.

**Files to transfer** (keep the folder names; run from `MAE/` on the source machine):
```bash
# 1) dataset (~17 GB): needs datasets/imagenet100/{train,val}
wormhole send datasets/imagenet100/train      # then the same for .../val  (or tar both first)
# 2) the three finished checkpoints (~1.4 GB each) - only checkpoint-last.pth is needed
wormhole send out_decay07/birdge_dataset_lr_0.00015_batch_size_64_min_decay_0.7/checkpoint-last.pth            # fixed baseline
wormhole send out_task2_learnable/birdge_dataset_lr_0.00015_batch_size_64_min_decay_0.7_learnable/checkpoint-last.pth
wormhole send out_task2_rank_depth/birdge_dataset_lr_0.00015_batch_size_64_min_decay_0.7_rank_depth/checkpoint-last.pth
```
On the receiving PC run `wormhole receive <code>` for each (receiving a single file keeps its name, so rename/put them into
`old_ckpts/fixed.pth`, `old_ckpts/learnable.pth`, `old_ckpts/rank_depth_v1.pth`). **Keep these OUT of `out_task2_*/`**: the training script resumes automatically from
`out_task2_<mode>/.../checkpoint-last.pth`, so a stale checkpoint there would skip the retraining.
`results/*.json` already travel with git.

**Order on the A6000** (needs `torch`, `timm`, `tqdm`; set `PY=...` / `DATA=...` if not the defaults `python` / `./datasets/imagenet100`):
```bash
# A. linear probes of the already-trained checkpoints (no retraining)
bash scripts/eval_ckpt.sh old_ckpts/fixed.pth          mae_vit_base_patch16 ./datasets/imagenet100 --min_decay 0.7
bash scripts/eval_ckpt.sh old_ckpts/learnable.pth      mae_vit_base_patch16 ./datasets/imagenet100 --min_decay 0.7 --decay_mode learnable
bash scripts/eval_ckpt.sh old_ckpts/rank_depth_v1.pth  mae_vit_base_patch16 ./datasets/imagenet100 --min_decay 0.7 --decay_mode rank   # old, buggy controller
# B. retrain the rank-targeted runs from scratch (fixed controller). ~6.5 GB each, so several fit on a 48 GB card; separate terminals/screens
bash scripts/run_task2.sh rank_depth --rank_gain 0.03 --rank_max_step 0.002 --rank_decay_floor 0.6
bash scripts/run_task2.sh rank_const
bash scripts/run_task2.sh rank_progress
bash scripts/run_task2.sh rank_depth_progress
# C. optional: extra seeds for the final table, each with its own --log_dir (e.g. --seed 1 --log_dir out_task2_rank_depth_s1)
```
B trains, then runs KNN/rank (`results/<tag>.json`) and the linear probe (`output_dir/eval_*/console.log`) automatically.
Fill the results table above from `results/*.json` and the probe logs.

## Tested
Controller unit test (4-block model, no autograd errors), a 3-epoch end-to-end smoke run (training, controller, resume, validation), `eval_task2.py` on smoke and trained checkpoints, and full 100-epoch runs of `rank_depth` and `learnable`.
