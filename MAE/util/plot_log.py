"""Plot training curves from <run_dir>/log.txt (JSON lines written by main_pretrain.py).
python -m util.plot_log RUN_DIR [RUN_DIR ...] --out curves.png   # several runs are overlaid for comparison"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt


def read_log(run_dir):
    rows = {}
    with open(os.path.join(run_dir, 'log.txt')) as f:
        for line in f:
            r = json.loads(line)
            rows[r['epoch']] = r  # resumed runs: keep the latest entry per epoch
    return [rows[e] for e in sorted(rows)]


def plot_log(run_dirs, out):
    panels = [('train_loss', 'loss'), ('lr_end', 'learning rate'), ('train_grad_norm', 'grad norm'),
              ('train_enc_feat_norm_0', 'encoder feature norm (last block)'), ('epoch_time_s', 'epoch time (s)'),
              ('max_mem_mb', 'peak GPU memory (MB)')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for run_dir in run_dirs:
        rows = read_log(run_dir)
        label = os.path.basename(os.path.normpath(run_dir))
        for ax, (key, title) in zip(axes.flat, panels):
            pts = [(r['epoch'], r[key]) for r in rows if key in r]
            if pts:
                ax.plot(*zip(*pts), marker='o', ms=3, label=label)
            ax.set_title(title)
            ax.set_xlabel('epoch')
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if len(run_dirs) > 1:
        axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run_dirs', nargs='+')
    p.add_argument('--out', default='training_curves.png')
    a = p.parse_args()
    plot_log(a.run_dirs, a.out)
