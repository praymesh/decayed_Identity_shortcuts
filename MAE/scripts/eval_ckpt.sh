#!/bin/bash
# Usage: bash scripts/eval_ckpt.sh CHECKPOINT MODEL DATASET [extra evaluation arguments]
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 CHECKPOINT MODEL DATASET [extra evaluation arguments]" >&2
  exit 2
fi
CK=$(realpath "$1")
MODEL=$2
DATA=$(realpath "$3")
shift 3
cd "$(dirname "$0")/.."
NAME=$(basename "$CK" .pth)
RUN_DIR="output_dir/eval_${NAME}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
# Capture failures as well as normal output. pipefail preserves torchrun's exit status.
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
trap 'status=$?; echo "$status" > "$RUN_DIR/exit_code.txt"' EXIT
echo "Results directory: $RUN_DIR"
torchrun --standalone --nproc_per_node 1 main_eval.py \
  --resume "$CK" --model "$MODEL" --data_path "$DATA" \
  --output_dir "$RUN_DIR" --norm_pix_loss --eval_only \
  --batch_size 8 --num_workers 4 "$@"
