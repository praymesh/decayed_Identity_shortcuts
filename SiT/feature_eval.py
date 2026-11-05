"""Label-free SiT feature extraction: frozen EMA at clean t=1 and null condition."""
from pathlib import Path

import torch
from torchvision import transforms
from diffusers import AutoencoderKL


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
