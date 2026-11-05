"""Cache frozen VAE posteriors for both pixel flips; resample latents during training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class LatentDataset(Dataset):
    def __init__(self, root, split, flip=False):
        self.moments = np.load(Path(root) / (split + '.npy'), mmap_mode='r')
        self.targets = np.load(Path(root) / (split + '_labels.npy')).tolist()
        self.flip = flip

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        view = int(torch.rand(()) < .5) if self.flip else 0
        return torch.from_numpy(self.moments[index, view].copy()), self.targets[index]


def prepare(args):
    from diffusers import AutoencoderKL
    from train import center_crop_arr
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(7 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    classes = (Path(__file__).resolve().parents[1] / 'MAE/configs/imagenet100_cmc_classes.txt').read_text().split()
    vae = AutoencoderKL.from_pretrained(args.vae, local_files_only=True, use_safetensors=True).cuda().eval().requires_grad_(False)
    metadata = dict(status='preparing', source=str(args.data.resolve()),
                    dataset={key: value for key, value in json.loads((args.data / 'verified.json').read_text()).items()
                             if key in ('repo', 'revision', 'resized_shorter_side', 'counts', 'class_to_idx')},
                    vae=json.loads((Path(__file__).parent / 'configs/vae.json').read_text()),
                    transform='ADM center crop 256; pixel-space horizontal flips encoded separately',
                    dtype='float32', channels='4 posterior mean + 4 log variance; scale latents by .18215 AFTER sampling',
                    train_flip_probability=.5, splits={})
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for split in ('train', 'val'):
        transform = transforms.Compose([
            transforms.Lambda(lambda image: center_crop_arr(image, 256)),
            transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
        dataset = ImageFolder(args.data / split, transform=transform)
        assert dataset.classes == classes and len(dataset) == (126689 if split == 'train' else 5000)
        views = 2 if split == 'train' else 1
        path = root / (split + '.npy')
        array = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(dataset), views, 8, 32, 32))
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False,
                            generator=torch.Generator().manual_seed(0), pin_memory=True)
        offset = 0
        with torch.inference_mode():
            for images, _ in loader:
                images = images.cuda(non_blocking=True)
                for view in range(views):
                    posterior = vae.encode(images if view == 0 else images.flip(-1)).latent_dist
                    values = torch.cat([posterior.mean, posterior.logvar], dim=1).float().cpu().numpy()
                    assert np.isfinite(values).all()
                    array[offset:offset+len(images), view] = values
                offset += len(images)
                if offset % 4096 == 0:
                    array.flush()
                    print(split, offset, '/', len(dataset), flush=True)
        array.flush()
        del array
        np.save(root / (split + '_labels.npy'), np.array(dataset.targets, dtype='int64'))
        with path.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        metadata['splits'][split] = dict(images=offset, views=views, bytes=path.stat().st_size, sha256=digest)
        metadata['class_to_idx'] = dataset.class_to_idx
        (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    metadata['status'] = 'verified'
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Verified latent cache:', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vae', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    prepare(parser.parse_args())
