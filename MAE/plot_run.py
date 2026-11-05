"""Plot recorded measurements for one run; never reruns training."""
import argparse
import csv
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


def compare_runs(roots, output):
    """Selected report figures and small tables from completed, separate run logs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs, summary, depths = [], [], []
    for root in map(Path, roots):
        metadata = json.loads((root / 'logs/settings.json').read_text())
        records = [json.loads(line) for line in (root / 'logs/metrics.jsonl').read_text().splitlines()]
        latest = lambda event: next(row for row in reversed(records) if row['event'] == event)
        if latest('status')['status'] != 'completed':
            raise ValueError(f'Run is not complete: {root}')
        name = metadata['method_name']
        layer = latest('depth_metrics')
        row = dict(method=name, run_id=root.name, **metadata['method'],
                   train_mae=latest('train_epoch')['mae_train_loss'],
                   validation_mae=latest('validation')['mae_val_loss'],
                   knn_percent=latest('knn')['accuracy_percent'],
                   linear_probe_percent=latest('linear_probe_validation')['accuracy_percent'],
                   optimizer_updates=latest('train_epoch')['optimizer_updates'],
                   run_wall_hours=latest('status')['elapsed_seconds'] / 3600,
                   peak_allocated_gib=max(r['peak_allocated_mib'] for r in records) / 1024,
                   peak_reserved_gib=max(r['peak_reserved_mib'] for r in records) / 1024,
                   final_cls_rank_entropy=layer['effective_rank'],
                   final_high_frequency_fraction=layer['high_frequency_fraction'],
                   final_total_variation=layer['total_variation'])
        summary.append(row)
        for record in records:
            if record['event'] == 'depth_metrics':
                depths.append(dict(method=name, layer=record['layer'], samples=record['samples'],
                                   rank_entropy=record['effective_rank'],
                                   high_frequency_fraction=record['high_frequency_fraction'],
                                   total_variation=record['total_variation'],
                                   spatial_energy=record['total_feature_energy'],
                                   high_frequency_energy=record['total_feature_energy'] * record['high_frequency_fraction'],
                                   low_frequency_energy=record['total_feature_energy'] * (1 - record['high_frequency_fraction'])))
        runs.append((name, records))
    for filename, rows in [('summary.csv', summary), ('depth_metrics.csv', depths)]:
        with (output / filename).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    title = 'ViT-B/16; 224 px input; CMC-160 ImageNet-100; 200 epochs; seed 0'
    colors = ['#52677a', '#d38124', '#1c8575']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, (key, label) in enumerate([('knn_percent', 'KNN'), ('linear_probe_percent', 'Linear probe')]):
        values = [r[key] for r in summary]
        axes[index].bar(range(len(summary)), values, color=colors[:len(summary)])
        axes[index].set(xticks=range(len(summary)), xticklabels=[r['method'].replace('_', '\n') for r in summary],
                        ylabel='Validation accuracy (%)', title=label, ylim=(0, 65))
        for x, value in enumerate(values):
            axes[index].text(x, value + 1, f'{value:.1f}%', ha='center')
    fig.suptitle(title + '\nFixed subsets: 1,000 training and 1,000 validation images')
    fig.tight_layout()
    fig.savefig(output / 'classification.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, (name, records) in enumerate(runs):
        for axis, event, key in [(axes[0], 'train_epoch', 'mae_train_loss'), (axes[1], 'validation', 'mae_val_loss')]:
            rows = [r for r in records if r['event'] == event]
            axis.plot([r['epoch'] + 1 for r in rows], [r[key] for r in rows], label=name, color=colors[index])
            axis.set(xlabel='MAE epoch', ylabel='Reconstruction loss')
    axes[0].set_title('Sample-weighted training epoch means')
    axes[1].set_title('Validation: all 5,000 images, fixed crops and masks')
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / 'mae_losses.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, key, label in zip(axes.flat,
            ['rank_entropy', 'high_frequency_fraction', 'total_variation',
             'low_frequency_energy', 'high_frequency_energy', 'spatial_energy'],
            ['CLS covariance entropy (no exponentiation)', 'High-frequency / non-DC energy',
             'Spatial variation (mean squared difference)', 'Low-frequency energy (feature units)',
             'High-frequency energy (feature units)', 'Total non-DC energy (feature units)']):
        for index, (name, _) in enumerate(runs):
            rows = [r for r in depths if r['method'] == name]
            axis.plot([r['layer'] for r in rows], [r[key] for r in rows], label=name, color=colors[index])
        axis.set(xlabel='Encoder block', ylabel=label)
        axis.grid(alpha=.2)
        if key.endswith('_energy'):
            axis.set_yscale('log')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + '\n200 complete unmasked validation images; radial cutoff 0.25 cycles/patch')
    fig.tight_layout()
    fig.savefig(output / 'depth_comparison.png', dpi=160)
    plt.close(fig)
    print('Saved comparison tables and figures to', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path, nargs='+')
    parser.add_argument('--compare_output', type=Path)
    args = parser.parse_args()
    if args.compare_output:
        compare_runs(args.run, args.compare_output)
    elif len(args.run) == 1:
        plot_run(args.run[0])
    else:
        parser.error('Multiple runs require --compare_output')
