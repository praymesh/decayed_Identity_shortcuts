"""Evaluate completed MAE/SiT queues without changing the trained backbones."""
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
    parser.add_argument('--mae-queue', type=Path, required=True)
    parser.add_argument('--sit-queue', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = args.output_dir.resolve()/f'queue-{stamp}-{uuid.uuid4().hex[:8]}'
    jobs = []
    for family, source in [('MAE', args.mae_queue), ('SiT', args.sit_queue)]:
        state = json.loads((source/'queue.json').read_text())
        if state['status'] != 'completed':
            raise ValueError(f'Source queue is incomplete: {source}')
        for job in state['jobs']:
            method = job['method']
            runs = [p for p in (source/method).iterdir() if (p/'checkpoints/checkpoint-last.pth').is_file()]
            if len(runs) != 1:
                raise ValueError(f'Expected exactly one completed checkpoint: {source/method}')
            script = code/family/('evaluate_checkpoint.py' if family=='MAE' else 'probe_layers.py')
            command = [sys.executable, '-u', str(script), '--checkpoint', str((runs[0]/'checkpoints/checkpoint-last.pth').resolve()),
                       '--output-dir', str(root/family/method), '--memory-budget-gib', '7']
            if family == 'SiT':
                command += ['--selection-file', str(root/'sit_selection.json')]
                if method == 'standard':
                    command.append('--choose-setting')
            jobs.append(dict(family=family, method=method, command=command, status='pending'))
            print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    (root/'logs').mkdir(parents=True)
    state = dict(status='running', queue=str(root), jobs=jobs)
    def save():
        temporary = root/'queue.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(root/'queue.json')
    save()
    print('Queue:', root, flush=True)
    for job in jobs:
        job.update(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); save()
        with (root/'logs'/f"{job['family']}-{job['method']}.txt").open('a', buffering=1) as console:
            result = subprocess.run(job['command'], cwd=code, env=dict(os.environ, OMP_NUM_THREADS='4'),
                                    stdout=console, stderr=subprocess.STDOUT)
        job.update(status='completed' if result.returncode==0 else 'failed', returncode=result.returncode,
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); save()
    state['status'] = 'completed' if all(job['status']=='completed' for job in jobs) else 'finished_with_errors'
    save()


if __name__ == '__main__':
    main()
