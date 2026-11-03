#!/bin/bash
# Sequential queue of all Task 2 runs (one 8GB GPU fits one run at a time). Each run resumes if restarted.
cd "$(dirname "$0")/.." || exit 1
for m in rank_depth learnable rank_const rank_progress rank_depth_progress; do
  bash scripts/run_task2.sh $m || echo "$m FAILED" >> task2_queue_failures.txt
done
