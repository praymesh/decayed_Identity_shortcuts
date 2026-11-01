"""Task 2 evaluation: KNN accuracy + effective rank (per block) + stored decay factors.

Single-process (no torchrun). Linear probing uses the existing main_eval.py / scripts/eval_ckpt.sh.
Usage:
  python eval_task2.py --ckpt <checkpoint-last.pth> --decay_mode rank --data_path ./datasets/imagenet100 \
      --out results/rank_depth.json
"""
import argparse, json, os, time
import torch, torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

import models_mae
from util.rank_control import effective_rank


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model', default='mae_vit_base_patch16')
    p.add_argument('--decay_mode', default='fixed', choices=['fixed', 'learnable', 'rank'])
    p.add_argument('--min_decay', default=0.7, type=float, help='only used to rebuild the fixed schedule')
    p.add_argument('--data_path', default='./datasets/imagenet100')
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--num_workers', default=8, type=int)
    p.add_argument('--knn_k', default=20, type=int)
    p.add_argument('--knn_T', default=0.07, type=float)
    p.add_argument('--train_subset', default=0, type=int, help='use only N train images (0 = all)')
    p.add_argument('--rank_tokens', default=2048, type=int)
    p.add_argument('--out', default='')
    return p.parse_args()


@torch.no_grad()
def extract(model, loader, dev, want_token_rank=False, rank_tokens=2048):
    """Returns per-block mean-pooled features [depth] x (N, C), labels, and (optionally) token-level erank fractions."""
    feats, labels, tok_rank = None, [], None
    for bi, (img, y) in enumerate(tqdm(loader, leave=False)):
        x = model.patch_embed(img.to(dev)) + model.pos_embed[:, 1:, :]
        x = torch.cat([(model.cls_token + model.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1), x], dim=1)
        outs = []
        with torch.autocast('cuda'):
            for blk in model.blocks:
                x = blk(x)
                outs.append(x.float())
        if feats is None:
            feats = [[] for _ in outs]
            tok_rank = [[] for _ in outs]
        for i, o in enumerate(outs):
            feats[i].append(o.mean(dim=1).cpu())
            if want_token_rank and bi < 8:  # first 8 batches are enough for a stable estimate
                r, m = effective_rank(o[:, 1:].reshape(-1, o.shape[-1]), rank_tokens)
                tok_rank[i].append((r / m).item())
        labels.append(y)
    feats = [torch.cat(f) for f in feats]
    tok = [sum(t) / len(t) for t in tok_rank] if want_token_rank else None
    return feats, torch.cat(labels), tok


@torch.no_grad()
def knn(train_f, train_y, val_f, val_y, k, T, n_cls, dev):
    train_f = F.normalize(train_f.to(dev), dim=1); val_f = F.normalize(val_f.to(dev), dim=1)
    train_y, val_y = train_y.to(dev), val_y.to(dev)
    correct = 0
    for i in range(0, val_f.shape[0], 500):
        sim, idx = (val_f[i:i + 500] @ train_f.T).topk(k, dim=1)
        w = (sim / T).exp()
        votes = torch.zeros(sim.shape[0], n_cls, device=dev).scatter_add_(1, train_y[idx], w)
        correct += (votes.argmax(1) == val_y[i:i + 500]).sum().item()
    return correct / val_f.shape[0]


def main():
    a = get_args()
    dev = torch.device('cuda')
    model = models_mae.__dict__[a.model](norm_pix_loss=True, min_decay=a.min_decay, decay_mode=a.decay_mode)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(ck['model'], strict=False)
    assert not [k for k in msg.missing_keys if k.startswith(('patch_embed.', 'blocks.', 'pos_embed', 'cls_token'))], msg
    model.to(dev).eval()

    tf = transforms.Compose([transforms.Resize(256, interpolation=3), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tr = datasets.ImageFolder(os.path.join(a.data_path, 'train'), tf)
    va = datasets.ImageFolder(os.path.join(a.data_path, 'val'), tf)
    if a.train_subset:
        g = torch.Generator().manual_seed(0)
        tr = torch.utils.data.Subset(tr, torch.randperm(len(tr), generator=g)[:a.train_subset].tolist())
    mk = lambda d: torch.utils.data.DataLoader(d, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False)

    t0 = time.time()
    tr_f, tr_y, _ = extract(model, mk(tr), dev)
    va_f, va_y, tok_rank = extract(model, mk(va), dev, want_token_rank=True, rank_tokens=a.rank_tokens)
    n_cls = len(va.classes)

    res = {'ckpt': a.ckpt, 'decay_mode': a.decay_mode, 'epoch': ck.get('epoch'),
           'knn_acc_per_block': [knn(tr_f[i], tr_y, va_f[i], va_y, a.knn_k, a.knn_T, n_cls, dev) for i in range(len(tr_f))],
           'erank_token_frac_per_block': tok_rank,                      # patch tokens, fraction of max rank
           'erank_pooled_per_block': [effective_rank(f, 5000)[0].item() for f in va_f],  # image-level features (n=5000 val)
           'decay_per_block': [b.decay.detach().float().cpu().tolist() if hasattr(b.decay, 'detach') else list(map(float, b.decay))
                               for b in model.blocks]}
    res['knn_acc'] = res['knn_acc_per_block'][-1]
    res['erank_pooled'] = res['erank_pooled_per_block'][-1]
    res['eval_time_s'] = time.time() - t0
    print(json.dumps({k: v for k, v in res.items() if k not in ('decay_per_block',)}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
