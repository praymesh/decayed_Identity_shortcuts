"""Sequential 200-epoch SiT shortcut comparisons with the verified latent cache."""
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
    code = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--latent-cache', type=Path, required=True)
    parser.add_argument('--model', default='SiT-B/4')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--memory-budget-gib', type=float, default=7)
    parser.add_argument('--results-dir', type=Path, default=code.parent / 'runs/sit')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = args.results_dir.resolve() / f'queue-{stamp}-{uuid.uuid4().hex[:8]}'
    common = [sys.executable, '-u', str(code / 'train.py'), '--local', '--model', args.model,
              '--data-path', str(args.data_root.resolve() / 'train'), '--latent-cache', str(args.latent_cache.resolve()),
              '--num-classes', '100', '--epochs', str(args.epochs), '--global-batch-size', '256',
              '--batch-size', str(args.batch_size), '--num-workers', str(args.num_workers),
              '--memory-budget-gib', str(args.memory_budget_gib), '--global-seed', '0',
              '--val-every', '5', '--feature-eval', '--probe-epochs', '200']
    jobs = []
    for name, decay, beta in [('standard', '1', '0'), ('fixed_decay', '.8', '0'),
                              ('frequency_selective', '.8', '.5')]:
        command = common + ['--min-sigma', decay, '--beta', beta, '--results-dir', str(root / name)]
        jobs.append(dict(method=name, command=command, status='pending'))
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    metadata = json.loads((args.latent_cache / 'metadata.json').read_text())
    if metadata['status'] != 'verified':
        raise ValueError('Latent cache is not ready')
    (root / 'logs').mkdir(parents=True)
    state = dict(status='running', family='SiT', model=args.model, queue=str(root), jobs=jobs)
    def save():
        temporary = root / 'queue.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(root / 'queue.json')
    save()
    print('Queue:', root, flush=True)
    for job in jobs:
        job.update(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        with (root / 'logs' / (job['method'] + '.txt')).open('a', buffering=1) as console:
            process = subprocess.run(job['command'], cwd=code.parent, env=dict(os.environ, OMP_NUM_THREADS='4'),
                                     stdout=console, stderr=subprocess.STDOUT)
        job.update(status='completed' if process.returncode == 0 else 'failed', returncode=process.returncode,
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
    state['status'] = 'completed' if all(j['status'] == 'completed' for j in jobs) else 'finished_with_errors'
    save()


if __name__ == '__main__':
    main()
