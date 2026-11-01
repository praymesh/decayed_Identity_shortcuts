#!/bin/bash
# Downloads ImageNet-100 (~15GB) from Kaggle into $DATA_DIR and reshapes it to
#   $DATA_DIR/{train,val}/<class>/*.JPEG   (what ImageFolder / main_eval.py expect)
# Needs: pip install kaggle ; and ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token)
set -e
DATA_DIR=${1:-$HOME/datasets/imagenet100}
RAW="$DATA_DIR/_raw"
mkdir -p "$RAW"
[ -n "$KAGGLE_API_TOKEN" ] || [ -f ~/.kaggle/kaggle.json ] || { echo "Set KAGGLE_API_TOKEN or create ~/.kaggle/kaggle.json"; exit 1; }
kaggle datasets download -d ambityga/imagenet100 -p "$RAW" --unzip
# the dataset ships train.X1..X4 (train split in 4 folders) + val.X
mkdir -p "$DATA_DIR/train" "$DATA_DIR/val"
for d in "$RAW"/train.X*; do cp -rl "$d"/* "$DATA_DIR/train/" 2>/dev/null || mv "$d"/* "$DATA_DIR/train/"; done
mv "$RAW"/val.X/* "$DATA_DIR/val/"
echo "classes: train=$(ls "$DATA_DIR/train" | wc -l) val=$(ls "$DATA_DIR/val" | wc -l)"
echo "images:  train=$(find "$DATA_DIR/train" -type f | wc -l) val=$(find "$DATA_DIR/val" -type f | wc -l)"
echo "Expect 100 classes, ~126689 train, 5000 val. If OK: rm -rf $RAW"
