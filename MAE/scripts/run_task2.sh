#!/bin/bash
# Task 2 (Rank-Targeted Residual Regulation): pretrain + evaluate.
# Usage:  bash scripts/run_task2.sh MODE [extra main_pretrain.py args]
#   MODE = rank_depth | rank_const | rank_progress | rank_depth_progress | learnable
# Same setting as the baseline run (ImageNet-100, ViT-B/16, 100 ep, bs 64 x accum 4, blr 1.5e-4, min_decay 0.7).
# Re-running resumes from checkpoint-last automatically.
set -e
cd "$(dirname "$0")/.."
# use the active env if there is one; fall back to the original machine's conda env
[ -d /home/dse_413/.conda/envs/mae/bin ] && export PATH=/home/dse_413/.conda/envs/mae/bin:$PATH
PY=${PY:-python}
DATA=${DATA:-./datasets/imagenet100}
MODE=${1:?MODE required}; shift || true
case $MODE in
  learnable) ARGS="--decay_mode learnable"; TAG=learnable ;;
  rank_*)    SCHED=${MODE#rank_}; ARGS="--decay_mode rank --rank_schedule $SCHED"; TAG=rank_$SCHED ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
LOG_DIR=out_task2_$TAG
$PY main_pretrain.py \
  --batch_size 64 --accum_iter 4 --epochs 100 --warmup_epochs 20 \
  --model mae_vit_base_patch16 --norm_pix_loss --min_decay 0.7 --blr 0.00015 \
  --num_workers 8 --data_path $DATA/train --log_dir $LOG_DIR --resume last $ARGS "$@" \
  2>&1 | tee -a ${LOG_DIR}_train.log
CK=$(ls -d $LOG_DIR/*/ | head -1)checkpoint-last.pth
DM=${ARGS#--decay_mode }; DM=${DM%% *}
# KNN + effective rank (per block) + final decays
$PY eval_task2.py --ckpt $CK --decay_mode $DM --data_path $DATA --out results/$TAG.json
# linear probe (existing protocol): output in output_dir/eval_*
bash scripts/eval_ckpt.sh $CK mae_vit_base_patch16 $DATA --min_decay 0.7 --decay_mode $DM
