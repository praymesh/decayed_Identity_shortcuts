"""Run folders and append-only records for MAE pretraining."""
import datetime
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)

    def flush(self):
        self.stream.flush()
        self.file.flush()


class RunLog:
    def __init__(self, root, args):
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.run_id = f'{run_id}-{uuid.uuid4().hex[:8]}'
        self.root = Path(root).resolve() / self.run_id
        for folder in ('logs', 'curves', 'checkpoints'):
            (self.root / folder).mkdir(parents=True, exist_ok=False)
        self.events = (self.root / 'logs/metrics.jsonl').open('a', buffering=1)
        self.console = (self.root / 'logs/console.txt').open('a', buffering=1)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(sys.stdout, self.console), Tee(sys.stderr, self.console)
        self.started = time.monotonic()
        repo = Path(__file__).resolve().parents[2]
        def git(*command):
            return subprocess.check_output(['git', '-C', str(repo), *command], text=True).strip()
        versions = {}
        for name in ('torch', 'torchvision', 'timm', 'numpy', 'matplotlib', 'flash-attn'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        metadata = dict(run_id=self.run_id, settings=vars(args).copy(),
                        command=shlex.join([sys.executable, *sys.argv]),
                        git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                        python=sys.version, platform=platform.platform(), versions=versions,
                        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                        cuda=torch.version.cuda, precision='fp16_amp' if args.temp5 == 1 else 'fp32',
                        attention='flash_attn_if_supported_else_pytorch_sdpa')
        self.metadata = metadata
        self.save_metadata()
        self.record('status', status='running', resume_checkpoint=args.resume or None)

    def save_metadata(self, **fields):
        self.metadata.update(fields)
        path = self.root / 'logs/settings.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.metadata, indent=2, default=str) + '\n')
        temporary.replace(path)

    def record(self, event, **fields):
        row = dict(event=event, run_id=self.run_id, elapsed_seconds=time.monotonic() - self.started,
                   **fields)
        if torch.cuda.is_available():
            row.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        self.events.write(json.dumps(row, allow_nan=False) + '\n')
        self.events.flush()

    def close(self, status, error=None):
        self.record('status', status=status, error=error)
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.events.close()
        self.console.close()
