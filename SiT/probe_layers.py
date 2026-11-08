"""Small SiT layer/time grid; select on held-out TRAIN images, then test once."""
import argparse
import json
from pathlib import Path
import sys

import torch

from feature_eval import extract_layers
from latent_data import LatentDataset
from models import SiT_models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MAE'))
from evaluate_features import balanced_indices, linear_probe
from plot_run import plot_run
from util.feature_metrics import knn_accuracy
from util.run_log import RunLog


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--selection-file', type=Path, required=True)
    parser.add_argument('--choose-setting', action='store_true')
    parser.add_argument('--probe-epochs', type=int, default=200)
    parser.add_argument('--memory-budget-gib', type=float, default=7)
    options = parser.parse_args()
    saved = torch.load(options.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    args = argparse.Namespace(**saved['args'])
    args.resume, args.probe_epochs = str(options.checkpoint.resolve()), options.probe_epochs
    args.optimizer_updates, args.probe_batch_size = saved['optimizer_updates'], 256
    args.layers, args.times = [3, 6, 9, 12], [1., .75, .5, .25]
    args.feature_val_samples, args.selection_per_class, args.official_validation_samples = 500, 5, 5000
    args.memory_budget_gib = options.memory_budget_gib
    log = RunLog(options.output_dir, args, loss_name='velocity')
    try:
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(args.memory_budget_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(args.val_seed)
        model = SiT_models[args.model](num_classes=100, input_size=32)
        model.set_method_settings(**saved['method_settings'])
        model.load_state_dict(saved['ema'], strict=True)
        model.cuda().eval().requires_grad_(False)
        versions = [p._version for p in model.parameters()]
        metadata = json.loads((Path(args.latent_cache)/'metadata.json').read_text())
        classes = (Path(__file__).resolve().parents[1]/'MAE/configs/imagenet100_cmc_classes.txt').read_text().split()
        if metadata['status'] != 'verified' or metadata['class_to_idx'] != dict(zip(classes, range(100))):
            raise ValueError('Unverified latent cache or class mapping')
        train, validation = LatentDataset(args.latent_cache, 'train'), LatentDataset(args.latent_cache, 'val')
        indices = balanced_indices(train.targets, 15, torch.Generator().manual_seed(args.val_seed))
        positions = torch.arange(1500).reshape(100, 15)
        fit, holdout = positions[:, :10].flatten(), positions[:, 10:].flatten()
        labels = torch.tensor(train.targets)[indices]
        log.save_metadata(dataset=dict(kind='ImageNet-100 SiT layer/noise audit', **metadata),
                          method=model.method_settings(), source_checkpoint=args.resume,
                          backbone_frozen=True, weights='EMA', pooling='Mean patch tokens; null class; cached posterior means',
                          fit_indices=[indices[i] for i in fit], selection_indices=[indices[i] for i in holdout],
                          selection_split='500 held-out training images; disjoint from 1000 probe-fit images',
                          precision='bf16_feature_autocast; fp32_CPU_classifier')
        cache = log.root/'feature_cache'; cache.mkdir()
        rows = []
        for time in args.times:
            features = extract_layers(model, train, indices, args.layers, time, args.val_seed)
            assert torch.isfinite(features).all()
            torch.save(dict(features=features, labels=labels, indices=indices), cache/f't-{time:g}.pt')
            for slot, layer in enumerate(args.layers):
                args.layer, args.noise_time = layer, time
                torch.manual_seed(args.val_seed)
                probe = RunLog(log.root/'probes'/f'layer-{layer}-t-{time:g}', args, loss_name='velocity')
                try:
                    probe.save_metadata(dataset=dict(kind='SiT TRAIN holdout, not official validation'),
                                        method=model.method_settings(), source_checkpoint=args.resume,
                                        feature_cache=str(cache/f't-{time:g}.pt'), backbone_frozen=True,
                                        precision='fp32_CPU_classifier', validation_split='training_holdout')
                    head = linear_probe(features[fit, slot], labels[fit], features[holdout, slot], labels[holdout],
                                        args, probe, saved['epoch'], pooling=f'Mean patch tokens, block {layer}, t={time}, null class')
                    with torch.no_grad():
                        accuracy = (head(features[holdout, slot]).argmax(1) == labels[holdout]).float().mean().item()*100
                    rows.append(dict(layer=layer, noise_time=time, selection_accuracy_percent=accuracy, run=str(probe.root)))
                    plot_run(probe.root)
                except BaseException as error:
                    probe.close('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', str(error)); raise
                else:
                    probe.close('completed')
                (log.root/'grid.json').write_text(json.dumps(rows, indent=2)+'\n')
        if options.choose_setting:
            if saved['method_settings']['min_sigma'] != 1 or options.selection_file.exists():
                raise ValueError('Only the standard control may create a new shared setting')
            choice = max(rows, key=lambda row: row['selection_accuracy_percent'])
            options.selection_file.write_text(json.dumps(dict(choice, selection_source='standard control training holdout',
                checkpoint=args.resume, seed=args.val_seed, fit_indices=[indices[i] for i in fit],
                selection_indices=[indices[i] for i in holdout]), indent=2)+'\n')
        selected = json.loads(options.selection_file.read_text())
        assert selected['fit_indices'] == [indices[i] for i in fit]
        assert selected['selection_indices'] == [indices[i] for i in holdout]
        choice = next(row for row in rows if row['layer']==selected['layer'] and row['noise_time']==selected['noise_time'])
        state = torch.load(Path(choice['run'])/f"checkpoints/linear-probe-epoch-{saved['epoch']:04d}.pth", weights_only=True)
        values = torch.load(cache/f"t-{choice['noise_time']:g}.pt", weights_only=True)['features'][fit, args.layers.index(choice['layer'])]
        head = torch.nn.Linear(values.shape[1], 100); head.load_state_dict(state['model']); head.eval()
        test = extract_layers(model, validation, list(range(len(validation))), [choice['layer']], choice['noise_time'], args.val_seed)[:, 0]
        targets = torch.tensor(validation.targets)
        with torch.no_grad():
            accuracy = (head(test).argmax(1)==targets).float().mean().item()*100
        torch.save(dict(train=values, labels=labels[fit], validation=test, targets=targets,
                        train_indices=[indices[i] for i in fit], validation_indices=list(range(len(validation))),
                        layer=choice['layer'], noise_time=choice['noise_time']), cache/'selected.pt')
        log.save_metadata(selected_setting=selected, official_validation_used_for_selection=False)
        log.record('linear_probe_validation', epoch=saved['epoch'], optimizer_updates=args.optimizer_updates,
                   probe_epoch=args.probe_epochs, accuracy_percent=accuracy, samples=len(targets), split='official_validation')
        log.record('knn', epoch=saved['epoch'], optimizer_updates=args.optimizer_updates,
                   accuracy_percent=knn_accuracy(values, labels[fit], test, targets), train_samples=len(fit),
                   samples=len(targets), k=20, temperature=.07, subset=True)
        assert versions == [p._version for p in model.parameters()]
        plot_run(log.root)
        print('Official validation accuracy:', accuracy, 'Shared setting:', selected['layer'], selected['noise_time'])
    except BaseException as error:
        log.close('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', str(error)); raise
    else:
        log.close('completed')


if __name__ == '__main__':
    main()
