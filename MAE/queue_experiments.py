"""Run the first three matched comparisons sequentially on one GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--memory_budget_gib', type=float, default=30)
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parents[1] / 'runs')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if args.epochs < 2:
        raise ValueError('Use at least two epochs with the one-epoch warmup')
    if args.batch_size < 1 or 256 % args.batch_size:
        raise ValueError('Physical batch must divide the matched effective batch of 256')
    code = Path(__file__).resolve().parent
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = args.output_dir.resolve() / f'queue-{stamp}-{uuid.uuid4().hex[:8]}'
    common = [sys.executable, '-u', str(code / 'main_pretrain.py'),
              '--model', 'mae_vit_base_patch16',
              '--data_path', str(args.data_root.resolve() / 'train'),
              '--val_path', str(args.data_root.resolve() / 'val'),
              '--class_list', str(code / 'configs/imagenet100_cmc_classes.txt'),
              '--dataset_id', 'clane9/imagenet-100@0519dc2f402a3a18c6e57f7913db059215eee25b',
              '--epochs', str(args.epochs), '--warmup_epochs', str(max(1, args.epochs // 10)),
              '--batch_size', str(args.batch_size), '--accum_iter', str(256 // args.batch_size), '--blr', '0.00015',
              '--weight_decay', '0.05', '--mask_ratio', '0.75', '--norm_pix_loss',
              '--seed', '0', '--num_workers', '4', '--memory_budget_gib', str(args.memory_budget_gib),
              '--eval_epochs_interval', '5',
              '--feature_eval', '--feature_train_per_class', '10',
              '--feature_val_samples', '1000', '--probe_epochs', '200']
    jobs = []
    for name, decay, beta in [('standard', '1', '0'), ('fixed_decay', '.7', '0'),
                              ('frequency_selective', '.7', '.5')]:
        command = common + ['--min_decay', decay, '--beta', beta, '--output_dir', str(root / name)]
        jobs.append(dict(method=name, command=command, status='pending'))
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not (args.data_root / 'verified.json').exists():
        raise ValueError('Prepare and verify the approved dataset first')
    (root / 'logs').mkdir(parents=True)
    state = dict(status='running', queue=str(root), jobs=jobs)
    def save():
        temporary = root / 'queue.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(root / 'queue.json')
    save()
    print('Queue:', root, flush=True)
    environment = dict(os.environ, OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    for job in jobs:
        job['status'] = 'running'
        job['started_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        with (root / 'logs' / (job['method'] + '.txt')).open('a', buffering=1) as console:
            result = subprocess.run(job['command'], cwd=code.parent, env=environment,
                                    stdout=console, stderr=subprocess.STDOUT)
        job['returncode'] = result.returncode
        job['status'] = 'completed' if result.returncode == 0 else 'failed'
        job['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    state['status'] = 'completed' if all(j['status'] == 'completed' for j in jobs) else 'finished_with_errors'
    save()


if __name__ == '__main__':
    main()
