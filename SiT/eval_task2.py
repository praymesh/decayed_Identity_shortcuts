"""Task 2 evaluation for SiT: KNN accuracy + linear probe + effective rank (per block) + stored decay factors.

Features = mean-pooled patch tokens of a block output, computed on VAE latents noised to time t
(SiT linear path: x_t = t*x_data + (1-t)*noise, so t=1 is clean data) with the *null* class label,
so no label information reaches the features. Single process (no torchrun).

  python eval_task2.py --ckpt results/<exp>/checkpoints/0100000.pt --model SiT-S/2 --decay_mode rank \
      --data_path ../MAE/datasets/imagenet100 --out results/rank_depth.json
Works for any --decay_mode (fixed / learnable / rank), so it is also used for the two baselines.
"""
import argparse, json, os, time
import torch, torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from models import SiT_models
from rank_control import effective_rank
from train import center_crop_arr


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model', default='SiT-S/2', choices=list(SiT_models.keys()))
    p.add_argument('--decay_mode', default='fixed', choices=['fixed', 'learnable', 'rank'])
    p.add_argument('--min_sigma', default=0.8, type=float, help='only used to rebuild the fixed schedule')
    p.add_argument('--num_classes', default=100, type=int, help='classes the model was trained with (null label = this index)')
    p.add_argument('--use_ema', type=int, default=1, help='evaluate the EMA weights stored in the checkpoint')
    p.add_argument('--t', default=0.8, type=float, help='noise level of the probe input (1 = clean)')
    p.add_argument('--data_path', default='../MAE/datasets/imagenet100')
    p.add_argument('--image_size', default=256, type=int)
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--num_workers', default=8, type=int)
    p.add_argument('--blocks', default='', help="comma list of block ids for KNN/probe (default: every 4th + last)")
    p.add_argument('--knn_k', default=20, type=int)
    p.add_argument('--knn_T', default=0.07, type=float)
    p.add_argument('--probe_epochs', default=30, type=int)
    p.add_argument('--train_subset', default=0, type=int, help='use only N train images (0 = all)')
    p.add_argument('--rank_tokens', default=2048, type=int)
    p.add_argument('--out', default='')
    return p.parse_args()


@torch.no_grad()
def extract(model, vae, loader, blocks, dev, t, num_classes, want_rank=False, rank_tokens=2048):
    """Per-block mean-pooled features (fp16, only `blocks`), labels, token-level erank fraction of every block."""
    feats, labels, tok_rank = {b: [] for b in blocks}, [], None
    g = torch.Generator(device=dev).manual_seed(0)  # same noise for every model -> comparable
    for bi, (img, y) in enumerate(tqdm(loader, leave=False)):
        img = img.to(dev)
        x1 = vae.encode(img).latent_dist.mean * 0.18215
        x0 = torch.randn(x1.shape, device=dev, generator=g)
        tt = torch.full((x1.shape[0],), t, device=dev)
        xt = t * x1 + (1 - t) * x0
        model.rank_collect = True
        model.rank_n_samples = xt.shape[0]
        with torch.autocast('cuda'):
            model(xt, tt, torch.full_like(y, num_classes).to(dev))  # null label
        outs = model.rank_feats
        model.rank_collect = False
        if tok_rank is None:
            tok_rank = [[] for _ in outs]
        for i, o in enumerate(outs):
            if i in feats:
                feats[i].append(o.float().mean(dim=1).half().cpu())
            if want_rank and bi < 8:
                r, m = effective_rank(o.reshape(-1, o.shape[-1]), rank_tokens)
                tok_rank[i].append((r / m).item())
        labels.append(y)
    feats = {b: torch.cat(f) for b, f in feats.items()}
    tok = [sum(t_) / len(t_) for t_ in tok_rank] if want_rank else None
    return feats, torch.cat(labels), tok


@torch.no_grad()
def knn(train_f, train_y, val_f, val_y, k, T_, n_cls, dev):
    train_f = F.normalize(train_f.float().to(dev), dim=1); val_f = F.normalize(val_f.float().to(dev), dim=1)
    train_y, val_y = train_y.to(dev), val_y.to(dev)
    correct = 0
    for i in range(0, val_f.shape[0], 500):
        sim, idx = (val_f[i:i + 500] @ train_f.T).topk(k, dim=1)
        votes = torch.zeros(sim.shape[0], n_cls, device=dev).scatter_add_(1, train_y[idx], (sim / T_).exp())
        correct += (votes.argmax(1) == val_y[i:i + 500]).sum().item()
    return correct / val_f.shape[0]


def linear_probe(train_f, train_y, val_f, val_y, n_cls, dev, epochs):
    """Frozen features -> standardise -> one linear layer (AdamW + cosine), top-1 on val."""
    tr, va = train_f.float().to(dev), val_f.float().to(dev)
    mu, sd = tr.mean(0, keepdim=True), tr.std(0, keepdim=True) + 1e-6
    tr, va = (tr - mu) / sd, (va - mu) / sd
    ty, vy = train_y.to(dev), val_y.to(dev)
    torch.manual_seed(0)
    lin = torch.nn.Linear(tr.shape[1], n_cls).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * ((tr.shape[0] + 1023) // 1024)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(epochs):
        perm = torch.randperm(tr.shape[0], device=dev)
        for i in range(0, tr.shape[0], 1024):
            idx = perm[i:i + 1024]
            loss = F.cross_entropy(lin(tr[idx]), ty[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return (lin(va).argmax(1) == vy).float().mean().item()


def main():
    a = get_args()
    dev = torch.device('cuda')
    model = SiT_models[a.model](input_size=a.image_size // 8, num_classes=a.num_classes, min_sigma=a.min_sigma, decay_mode=a.decay_mode)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['ema'] if a.use_ema and 'ema' in ck else ck.get('model', ck))
    model.to(dev).eval()
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema').to(dev).eval()
    depth = len(model.blocks)
    blocks = [int(b) for b in a.blocks.split(',')] if a.blocks else sorted(set(range(3, depth, 4)) | {depth - 1})

    tf = T.Compose([T.Lambda(lambda im: center_crop_arr(im, a.image_size)), T.ToTensor(), T.Normalize([.5] * 3, [.5] * 3)])
    tr = datasets.ImageFolder(os.path.join(a.data_path, 'train'), tf)
    va = datasets.ImageFolder(os.path.join(a.data_path, 'val'), tf)
    if a.train_subset:
        g = torch.Generator().manual_seed(0)
        tr = torch.utils.data.Subset(tr, torch.randperm(len(tr), generator=g)[:a.train_subset].tolist())
    mk = lambda d: torch.utils.data.DataLoader(d, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False)
    n_cls = len(va.classes)

    t0 = time.time()
    tr_f, tr_y, _ = extract(model, vae, mk(tr), blocks, dev, a.t, a.num_classes)
    va_f, va_y, tok_rank = extract(model, vae, mk(va), blocks, dev, a.t, a.num_classes, want_rank=True, rank_tokens=a.rank_tokens)
    res = {'ckpt': a.ckpt, 'model': a.model, 'decay_mode': a.decay_mode, 't': a.t, 'blocks': blocks,
           'knn_acc_per_block': {b: knn(tr_f[b], tr_y, va_f[b], va_y, a.knn_k, a.knn_T, n_cls, dev) for b in blocks},
           'probe_acc_per_block': {b: linear_probe(tr_f[b], tr_y, va_f[b], va_y, n_cls, dev, a.probe_epochs) for b in blocks},
           'erank_token_frac_per_block': tok_rank,  # patch tokens of every block, fraction of max rank
           'erank_pooled_per_block': {b: effective_rank(va_f[b].float(), 5000)[0].item() for b in blocks},
           'decay_per_block': [b.decay.detach().float().cpu().tolist() if hasattr(b.decay, 'detach') else list(map(float, b.decay))
                               for b in model.blocks]}
    res['knn_acc_best'] = max(res['knn_acc_per_block'].values())
    res['probe_acc_best'] = max(res['probe_acc_per_block'].values())
    res['eval_time_s'] = time.time() - t0
    print(json.dumps({k: v for k, v in res.items() if k != 'decay_per_block'}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
