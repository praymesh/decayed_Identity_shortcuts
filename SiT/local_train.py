"""Single-GPU SiT training using the shared run logger and update/validation loops."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import signal
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, DistributedSampler

from models import SiT_models
from transport import create_transport
from latent_data import LatentDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MAE'))
from engine_pretrain import train_one_epoch, evaluate
from util import misc
from util.run_log import RunLog


def train_local(args, update_ema, requires_grad, center_crop):
    args.seed, args.temp5 = args.global_seed, 1
    args.lr, args.lr_schedule, args.loss_name = 1e-4, 'constant', 'velocity'
    args.accum_iter = args.global_batch_size // args.batch_size
    args.val_seed, args.optimizer_updates, args.start_epoch = 12345, 0, 0
    args.input_size, args.mask_ratio, args.resume = args.image_size, 0, args.ckpt or ''
    args.weight_decay = 0
    args.feature_train_per_class, args.feature_val_samples = 10, 1000
    if args.batch_size < 1 or args.global_batch_size % args.batch_size or args.num_classes != 100:
        raise ValueError('Physical batch must divide effective batch; this runner uses the agreed 100 classes')
    if args.image_size != 256 or args.val_every < 1:
        raise ValueError('The verified cache uses 256-pixel images; validation interval must be positive')
    if not args.latent_cache:
        raise ValueError('Prepare --latent-cache first; its VAE and both pixel flips are recorded')
    log = RunLog(args.results_dir, args, loss_name='velocity')
    args.output_dir = str(log.root / 'checkpoints')
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    print('Run folder:', log.root)
    try:
        device = torch.device('cuda', 0)
        torch.cuda.set_per_process_memory_fraction(args.memory_budget_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        cache = Path(args.latent_cache)
        metadata = json.loads((cache / 'metadata.json').read_text())
        if metadata['status'] != 'verified':
            raise ValueError('Latent cache is incomplete')
        classes = (Path(__file__).resolve().parents[1] / 'MAE/configs/imagenet100_cmc_classes.txt').read_text().split()
        if sorted(metadata['class_to_idx']) != classes:
            raise ValueError('Latent cache class list differs from the agreed dataset')
        train = LatentDataset(cache, 'train', flip=True)
        validation = LatentDataset(cache, 'val')
        selection = dict(train_per_class=args.train_per_class, val_samples=args.val_samples)
        if args.train_per_class:
            indices = []
            generator = np.random.default_rng(args.seed)
            targets = np.asarray(train.targets)
            for label in range(100):
                choices = np.flatnonzero(targets == label)
                indices.extend(generator.choice(choices, args.train_per_class, replace=False).tolist())
            train = Subset(train, indices)
            selection['train_indices'] = indices
        if args.val_samples:
            indices = np.linspace(0, len(validation)-1, min(len(validation), args.val_samples), dtype=int).tolist()
            validation = Subset(validation, indices)
            selection['val_indices'] = indices
        args.dataset_fingerprint = hashlib.sha256((json.dumps(metadata, sort_keys=True) + json.dumps(selection, sort_keys=True)).encode()).hexdigest()
        generator = torch.Generator()
        sampler = DistributedSampler(train, num_replicas=1, rank=0, shuffle=True, seed=args.seed)
        loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True, generator=generator)
        val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.num_workers,
                                generator=torch.Generator().manual_seed(args.val_seed))
        if not len(loader):
            raise ValueError('Training loader is empty')
        model = SiT_models[args.model](num_classes=100, input_size=32, min_sigma=args.min_sigma, beta=args.beta).to(device)
        ema = deepcopy(model).eval()
        requires_grad(ema, False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
        scaler = misc.NativeScalerWithGradNormCount()
        saved = misc.load_model(args, model, optimizer, scaler)
        if saved:
            ema.load_state_dict(saved['ema'])
            ema.set_method_settings(**saved['method_settings'])
        transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
        def loss_fn(network, moments, labels):
            with torch.autocast('cuda', enabled=False):
                mean, logvar = moments.float().chunk(2, dim=1)
                latents = (mean + torch.exp(.5 * logvar) * torch.randn_like(mean)) * .18215
            return transport.training_losses(network, latents, dict(y=labels))['loss'].mean()
        log.save_metadata(settings=vars(args).copy(), method=model.method_settings(),
                          dataset=dict(kind='ImageNet-100 latent posteriors', **metadata, selection=selection),
                          effective_batch_size=args.global_batch_size,
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          validation_weights='EMA', feature_protocol='EMA; posterior mean; t=1; null class; mean tokens',
                          vae_cached=True, class_conditioning='100 training labels; null label during feature extraction')
        for epoch in range(args.start_epoch, args.epochs):
            sampler.set_epoch(epoch); generator.manual_seed(args.seed + epoch)
            stats = train_one_epoch(model, loader, optimizer, device, epoch, scaler, log, args,
                                    loss_fn=loss_fn, on_update=lambda: update_ema(ema, model))
            log.record('train_epoch', epoch=epoch, optimizer_updates=args.optimizer_updates,
                       velocity_train_loss=stats['loss'], lr=stats['lr'])
            if (epoch+1) % args.val_every == 0 or epoch+1 == args.epochs:
                evaluate(ema, val_loader, device, args, epoch, log, loss_fn=loss_fn)
                from plot_local import plot_local
                plot_local(log.root)
            misc.save_model(args, epoch, model, model, optimizer, scaler, save_last=True,
                            extra_state=dict(ema=ema.state_dict()))
            if epoch+1 == args.epochs:
                (log.root / 'checkpoints' / f'checkpoint-{epoch:05d}.pth').hardlink_to(
                    log.root / 'checkpoints/checkpoint-last.pth')
        if args.feature_eval:
            from feature_eval import feature_eval
            feature_eval(ema, args, log, device, center_crop)
        from plot_local import plot_local
        plot_local(log.root)
    except BaseException as error:
        log.close('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', str(error))
        raise
    else:
        log.close('completed')
