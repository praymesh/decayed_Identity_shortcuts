#!/bin/bash
# Task 2 on SiT (Rank-Targeted Residual Regulation): train + evaluate.
# Usage:  bash scripts/run_task2.sh MODE [extra train.py args]
#   MODE = fixed | learnable | rank_depth | rank_const | rank_progress | rank_depth_progress
# Env knobs: MODEL (default SiT-S/2), DATA, EPOCHS, NGPU, BS (global batch), WANDB=1 to enable wandb.
# fixed / learnable are the two baselines; the rank_* modes are Task 2.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
MODEL=${MODEL:-SiT-S/2}
DATA=${DATA:-../MAE/datasets/imagenet100}
EPOCHS=${EPOCHS:-100}
NGPU=${NGPU:-1}
BS=${BS:-256}
MODE=${1:?MODE required}; shift || true
case $MODE in
  fixed)     ARGS="--decay-mode fixed";     TAG=fixed ;;
  learnable) ARGS="--decay-mode learnable"; TAG=learnable ;;
  rank_*)    ARGS="--decay-mode rank --rank-schedule ${MODE#rank_}"; TAG=$MODE ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
DM=$(echo $ARGS | awk '{print $2}')
RES=results_task2/$TAG
[ "${WANDB:-0}" = 1 ] || ARGS="$ARGS --wandb"   # NB: in train.py the --wandb flag *disables* wandb
torchrun --nnodes=1 --nproc_per_node=$NGPU train.py --model $MODEL --data-path $DATA/train --num-classes 100 \
  --epochs $EPOCHS --global-batch-size $BS --results-dir $RES $ARGS "$@" 2>&1 | tee -a $RES.log
CK=$(ls -t $RES/*/checkpoints/*.pt | head -1)
$PY eval_task2.py --ckpt $CK --model $MODEL --decay_mode $DM --data_path $DATA --out results_task2/$TAG.json
