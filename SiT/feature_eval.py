"""Label-free SiT feature extraction: frozen EMA at clean t=1 and null condition."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from diffusers import AutoencoderKL


@torch.no_grad()
def extract_layers(model, dataset, indices, layers, noise_time, seed, batch_size=32):
    """Cached posterior means, fixed noise, null labels; collect several layers once."""
    if not 0 <= noise_time <= 1 or any(layer < 1 or layer > len(model.enc_blocks)+len(model.dec_blocks) for layer in layers):
        raise ValueError('Invalid interpolant time or layer')
    values = [[] for _ in layers]
    blocks = list(model.enc_blocks) + list(model.dec_blocks)
    hooks = [blocks[layer-1].register_forward_hook(
        lambda module, inputs, output, slot=slot: values[slot].append(output.float().mean(1).cpu()))
        for slot, layer in enumerate(layers)]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=2,
                        generator=torch.Generator().manual_seed(seed))
    device = next(model.parameters()).device
    try:
        for moments, _ in loader:
            means = moments[:, :4].float() * .18215
            noise = torch.randn(means.shape, generator=generator)
            inputs = (noise_time * means + (1-noise_time) * noise).to(device)
            times = torch.full((len(inputs),), noise_time, device=device)
            labels = torch.full((len(inputs),), model.y_embedder.num_classes, device=device, dtype=torch.long)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                model.forward_tokens(inputs, times, labels)
        return torch.stack([torch.cat(layer) for layer in values], dim=1)
    finally:
        for hook in hooks:
            hook.remove()


class FeatureAdapter(torch.nn.Module):
    feature_has_cls = False
    pooling_description = 'Mean final SiT patch tokens; EMA; t=1; VAE posterior mean; null class'

    def __init__(self, model, vae):
        super().__init__()
        self.model, self.vae = model, vae

    @property
    def blocks(self):
        return list(self.model.enc_blocks) + list(self.model.dec_blocks)

    @property
    def patch_embed(self):
        return self.model.x_embedder

    def forward_feat(self, images):
        with torch.autocast('cuda', enabled=False):
            latents = self.vae.encode(images.float()).latent_dist.mode() * .18215
        times = latents.new_ones(len(latents))
        labels = torch.full((len(latents),), self.model.y_embedder.num_classes, device=latents.device, dtype=torch.long)
        return [self.model.forward_tokens(latents, times, labels)[0]]


def feature_eval(model, args, log, device, center_crop):
    from evaluate_features import evaluate_features
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True, use_safetensors=True).to(device).eval().requires_grad_(False)
    adapter = FeatureAdapter(model, vae).eval()
    transform = transforms.Compose([transforms.Lambda(lambda image: center_crop(image, 256)),
                                    transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
    args.feature_train_per_class, args.feature_val_samples = 10, 1000
    args.val_path = str(Path(args.data_path).resolve().parent / 'val')
    evaluate_features(adapter, transform, device, args, log, args.epochs-1)
