"""Frozen MAE checkpoint evaluation; zero subset limits mean all images."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

import models_mae
from evaluate_features import evaluate_features
from plot_run import plot_run
from util.run_log import RunLog


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--probe-epochs', type=int, default=200)
    parser.add_argument('--probe-batch-size', type=int, default=1024)
    parser.add_argument('--extraction-batch-size', type=int, default=8)
    parser.add_argument('--train-per-class', type=int, default=0)
    parser.add_argument('--val-samples', type=int, default=0)
    parser.add_argument('--memory-budget-gib', type=float, default=7)
    options = parser.parse_args()
    saved = torch.load(options.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    args = argparse.Namespace(**saved['args'])
    args.resume = str(options.checkpoint.resolve())
    args.probe_epochs, args.probe_batch_size = options.probe_epochs, options.probe_batch_size
    args.extraction_batch_size = options.extraction_batch_size
    args.feature_train_per_class, args.feature_val_samples = options.train_per_class, options.val_samples
    args.memory_budget_gib, args.optimizer_updates = options.memory_budget_gib, saved['optimizer_updates']
    log = RunLog(options.output_dir, args)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(args.val_seed); np.random.seed(args.val_seed); random.seed(args.val_seed)
        device = torch.device('cuda', 0)
        torch.cuda.set_per_process_memory_fraction(args.memory_budget_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats()
        dataset = json.loads((Path(args.data_path).parent / 'verified.json').read_text())
        if args.dataset_id != f"{dataset['repo']}@{dataset['revision']}":
            raise ValueError('Checkpoint and dataset versions differ')
        for split in (args.data_path, args.val_path):
            if ImageFolder(split).class_to_idx != dataset['class_to_idx']:
                raise ValueError('Dataset label mapping differs from verified ImageNet-100')
        model = models_mae.__dict__[args.model](img_size=args.input_size, norm_pix_loss=args.norm_pix_loss)
        model.set_method_settings(**saved['method_settings'])
        model.load_state_dict(saved['model'], strict=True)
        model.to(device).eval().requires_grad_(False)
        versions = [p._version for p in model.parameters()]
        log.save_metadata(settings=vars(args), method=model.method_settings(),
                          dataset=dict(kind='ImageNet-100 frozen probe', **dataset),
                          source_checkpoint=args.resume, backbone_frozen=True,
                          precision=f'{torch.get_autocast_dtype("cuda")}_feature_autocast; fp32_CPU_classifier',
                          evaluation_subset=bool(options.train_per_class or options.val_samples),
                          pooling='Mean final raw tokens INCLUDING CLS; unchanged repository protocol')
        transform = transforms.Compose([transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.input_size), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        evaluate_features(model, transform, device, args, log, saved['epoch'])
        assert versions == [p._version for p in model.parameters()]
        plot_run(log.root)
    except BaseException as error:
        log.close('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', str(error))
        raise
    else:
        log.close('completed')


if __name__ == '__main__':
    main()
