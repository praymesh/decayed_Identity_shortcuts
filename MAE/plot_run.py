"""Plot recorded measurements for one run; never reruns training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_run(root):
    root = Path(root)
    records = [json.loads(line) for line in (root / 'logs/metrics.jsonl').read_text().splitlines()]
    settings = json.loads((root / 'logs/settings.json').read_text())
    method = settings.get('method', {})
    dataset = settings.get('dataset', {}).get('kind', 'unknown data')
    title = f'{root.name}\n{dataset}, a_min={method.get("min_decay")}, beta={method.get("beta")}'
    curves = root / 'curves'
    curves.mkdir(exist_ok=True)

    def plot(filename, ylabel, series, xlabel='Optimizer updates'):
        fig, axis = plt.subplots(figsize=(8, 4))
        found = False
        for key, label, event, xkey in series:
            rows = [r for r in records if r['event'] == event and r.get(key) is not None]
            if rows:
                axis.plot([r[xkey] for r in rows], [r[key] for r in rows],
                          label=label, marker='o' if len(rows) < 30 else None, markersize=3)
                found = True
        if found:
            axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
            axis.legend()
            axis.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(curves / filename, dpi=150)
        plt.close(fig)

    plot('mae_loss.png', 'MAE reconstruction loss', [
        ('mae_train_loss', 'Training', 'train_update', 'optimizer_updates'),
        ('mae_val_loss', 'Validation', 'validation', 'optimizer_updates')])
    plot('learning_rate.png', 'Learning rate', [('lr', 'Learning rate', 'train_update', 'optimizer_updates')])
    plot('linear_probe_loss.png', 'Classifier cross-entropy loss', [
        ('classifier_loss', 'Probe training', 'linear_probe', 'probe_epoch')], 'Probe epoch')
    plot('linear_probe_accuracy.png', 'Validation accuracy (%)', [
        ('accuracy_percent', 'Linear probe', 'linear_probe_validation', 'probe_epoch')], 'Probe epoch')
    plot('knn_accuracy.png', 'Validation accuracy (%)', [
        ('accuracy_percent', 'KNN', 'knn', 'optimizer_updates')])
    for key, label in [('effective_rank', 'Covariance singular-value entropy'),
                       ('high_frequency_fraction', 'High-frequency energy / total non-DC energy'),
                       ('total_variation', 'Mean squared spatial difference')]:
        plot(key + '.png', label, [(key, label, 'depth_metrics', 'layer')], 'Encoder block (1-based)')
    print('Saved available measurements to', curves)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    plot_run(parser.parse_args().run)
