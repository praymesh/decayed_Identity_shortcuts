"""Make a small symlinked subset of an ImageFolder dataset for fast dev runs.
python scripts/make_subset.py SRC DST --per_class 100"""
import argparse, os, random
p = argparse.ArgumentParser()
p.add_argument('src'); p.add_argument('dst'); p.add_argument('--per_class', type=int, default=100)
a = p.parse_args()
random.seed(0)
for split in ('train', 'val'):
    for c in sorted(os.listdir(f'{a.src}/{split}')):
        fs = sorted(os.listdir(f'{a.src}/{split}/{c}'))
        if split == 'train': fs = random.sample(fs, min(a.per_class, len(fs)))
        os.makedirs(f'{a.dst}/{split}/{c}', exist_ok=True)
        for f in fs:
            os.symlink(os.path.abspath(f'{a.src}/{split}/{c}/{f}'), f'{a.dst}/{split}/{c}/{f}')
