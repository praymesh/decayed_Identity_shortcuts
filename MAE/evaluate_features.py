"""Small-batch unmasked extraction, CPU probes, and depth measurements."""
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from util.feature_metrics import effective_rank, spatial_energy, knn_accuracy


@torch.no_grad()
def extract(model, dataset, device, seed):
    loader = DataLoader(dataset, batch_size=2, num_workers=2,
                        generator=torch.Generator().manual_seed(seed))
    features, labels = [], []
    for images, targets in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            tokens = model.forward_feat(images.to(device))[0]
        # Preserve util.misc.eval_model: mean over every final token, including CLS.
        features.append(tokens.float().mean(dim=1).cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def linear_probe(train, labels, validation, targets, args, log, epoch,
                 pooling='Final raw tokens averaged including CLS'):
    classifier = torch.nn.Linear(train.shape[1], 100)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=.01)
    probe_updates = 0
    for probe_epoch in range(args.probe_epochs):
        order = torch.randperm(len(train))
        total = 0.0
        for indices in order.split(256):
            loss = torch.nn.functional.cross_entropy(classifier(train[indices]), labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probe_updates += 1
            log.record('linear_probe_update', epoch=epoch, optimizer_updates=args.optimizer_updates,
                       probe_epoch=probe_epoch + 1, probe_updates=probe_updates,
                       classifier_loss=loss.item(), lr=optimizer.param_groups[0]['lr'], samples=len(indices))
            total += loss.item() * len(indices)
        log.record('linear_probe', epoch=epoch, optimizer_updates=args.optimizer_updates,
                   probe_epoch=probe_epoch + 1, classifier_loss=total / len(train),
                   train_samples=len(train))
        # Match the custom probe's AdamW and cosine schedule.
        optimizer.param_groups[0]['lr'] = .01 * .5 * (1 + math.cos(math.pi * probe_epoch / args.probe_epochs))
        if probe_epoch % 50 == 0 or probe_epoch + 1 == args.probe_epochs:
            with torch.no_grad():
                accuracy = (classifier(validation).argmax(1) == targets).float().mean().item() * 100
            log.record('linear_probe_validation', epoch=epoch, optimizer_updates=args.optimizer_updates,
                       probe_epoch=probe_epoch + 1, accuracy_percent=accuracy, samples=len(targets))
    print(f'Linear-probe accuracy: {accuracy:.2f}% ({len(targets)} validation images)')
    torch.save(dict(model=classifier.state_dict(), probe_updates=probe_updates,
                    pooling=pooling),
               log.root / 'checkpoints' / f'linear-probe-epoch-{epoch:04d}.pth')


@torch.no_grad()
def measure_depth(model, dataset, device, args, log, epoch):
    cls = [[] for _ in model.blocks]
    sums = [torch.zeros(3, dtype=torch.float64) for _ in model.blocks]
    hooks = []
    def capture(layer):
        def hook(module, inputs, output):
            tokens = output.detach().float().cpu()
            has_cls = getattr(model, 'feature_has_cls', True)
            cls[layer].append(tokens[:, 0] if has_cls else tokens.mean(1))
            patches = tokens[:, 1:] if has_cls else tokens
            energy, high, variation = spatial_energy(patches, model.patch_embed.grid_size)
            sums[layer] += torch.tensor([energy.sum(), high.sum(), variation.sum()], dtype=torch.float64)
        return hook
    try:
        for layer, block in enumerate(model.blocks):
            hooks.append(block.register_forward_hook(capture(layer)))
        extract(model, dataset, device, args.val_seed)
    finally:
        for hook in hooks:
            hook.remove()
    for layer, (features, total) in enumerate(zip(cls, sums)):
        log.record('depth_metrics', epoch=epoch, optimizer_updates=args.optimizer_updates,
                   layer=layer + 1, samples=len(dataset),
                   feature=('CLS' if getattr(model, 'feature_has_cls', True) else 'Mean patch tokens') + '; sample-by-sample covariance across channels',
                   effective_rank=effective_rank(torch.cat(features)),
                   frequency_cutoff_cycles_per_patch=.25,
                   high_frequency_fraction=(total[1] / total[0]).item() if total[0] else 0.0,
                   total_feature_energy=(total[0] / len(dataset)).item(),
                   total_variation=(total[2] / len(dataset)).item())


def evaluate_features(model, transform, device, args, log, epoch):
    was_training = model.training
    devices = [device.index] if device.type == 'cuda' else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.val_seed)
            train = ImageFolder(args.data_path, transform=transform)
            validation = ImageFolder(args.val_path or str(Path(args.data_path).parent / 'val'), transform=transform)
            train_indices = []
            for label in range(100):
                choices = [i for i, value in enumerate(train.targets) if value == label]
                order = torch.randperm(len(choices))[:args.feature_train_per_class].tolist()
                train_indices.extend(choices[i] for i in order)
            val_indices = torch.linspace(0, len(validation) - 1, min(args.feature_val_samples, len(validation))).long().tolist()
            train_features, labels = extract(model, Subset(train, train_indices), device, args.val_seed)
            val_features, targets = extract(model, Subset(validation, val_indices), device, args.val_seed)
            cache = log.root / 'feature_cache'
            cache.mkdir(exist_ok=True)
            torch.save(dict(train=train_features, labels=labels, validation=val_features, targets=targets,
                            train_indices=train_indices, validation_indices=val_indices,
                            pooling=getattr(model, 'pooling_description', 'Final raw tokens averaged including CLS')),
                       cache / f'epoch-{epoch:04d}.pt')
            accuracy = knn_accuracy(train_features, labels, val_features, targets)
            log.record('knn', epoch=epoch, optimizer_updates=args.optimizer_updates,
                       accuracy_percent=accuracy, train_samples=len(labels), samples=len(targets),
                       k=min(20, len(labels)), temperature=.07, subset=True)
            print(f'KNN accuracy: {accuracy:.2f}% ({len(targets)} validation images)')
            depth_indices = torch.linspace(0, len(val_indices) - 1, min(200, len(val_indices))).long().tolist()
            measure_depth(model, Subset(validation, [val_indices[i] for i in depth_indices]), device, args, log, epoch)
            if args.probe_epochs:
                linear_probe(train_features, labels, val_features, targets, args, log, epoch,
                             pooling=getattr(model, 'pooling_description', 'Final raw tokens averaged including CLS'))
    finally:
        model.train(was_training)
