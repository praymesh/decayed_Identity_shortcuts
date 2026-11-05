"""SiT curves from saved records; feature plots reuse the existing plotting helper."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_local(root):
    root = Path(root)
    from plot_run import plot_run
    plot_run(root)
    rows = [json.loads(line) for line in (root / 'logs/metrics.jsonl').read_text().splitlines()]
    settings = json.loads((root / 'logs/settings.json').read_text())
    for filename, series, ylabel in [
        ('velocity_loss.png', [('train_update', 'velocity_train_loss', 'Training'),
                               ('validation', 'velocity_val_loss', 'EMA validation')], 'Velocity prediction MSE'),
        ('learning_rate.png', [('train_update', 'lr', 'Learning rate')], 'Learning rate')]:
        fig, axis = plt.subplots(figsize=(8, 4))
        for event, key, label in series:
            data = [r for r in rows if r['event'] == event]
            if data:
                axis.plot([r['optimizer_updates'] for r in data], [r[key] for r in data], label=label)
        axis.set(xlabel='Optimizer updates', ylabel=ylabel,
                 title=f"{root.name}\n{settings['settings']['model']}, {settings.get('method')}")
        axis.legend(); axis.grid(alpha=.2); fig.tight_layout()
        fig.savefig(root / 'curves' / filename, dpi=150)
        plt.close(fig)
