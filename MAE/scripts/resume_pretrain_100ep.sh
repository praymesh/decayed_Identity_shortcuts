#!/bin/bash
# Resume MAE pretraining from checkpoint-last, 100-epoch schedule, wandb on.
cd "$(dirname "$0")/.." || exit 1
pkill -f "python.*main_pretrain.py"; sleep 5
cp out_decay07_train.log out_decay07_train_prev.log 2>/dev/null
setsid nohup /home/dse_413/.conda/envs/mae/bin/python main_pretrain.py \
  --batch_size 64 --accum_iter 4 --epochs 100 --warmup_epochs 20 \
  --model mae_vit_base_patch16 --norm_pix_loss --min_decay 0.7 --blr 0.00015 \
  --num_workers 8 --data_path ./datasets/imagenet100/train \
  --log_dir out_decay07 --resume last --wandb \
  > out_decay07_train.log 2>&1 < /dev/null &
