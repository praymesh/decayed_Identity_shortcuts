"""Download the approved CMC-160 release and verify it before making ImageFolder data."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import urllib.request

import pyarrow.parquet as pq
from PIL import Image

REPO = 'clane9/imagenet-100'
REVISION = '0519dc2f402a3a18c6e57f7913db059215eee25b'
BASE = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'


def read_url(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--destination', type=Path, default=Path('/tmp/imagenet100-cmc-160'))
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = Path(__file__).resolve().parent / 'configs'
    api = json.loads(read_url(f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true'))
    shards = [p for p in api['siblings'] if p['rfilename'].endswith('.parquet')]
    source_tree = ast.parse(read_url(BASE + 'scripts/classes.py').decode())
    names = next(ast.literal_eval(node) for node in ast.walk(source_tree)
                 if isinstance(node, ast.Dict) and len(node.keys) == 100)
    cmc = read_url('https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt').decode().split()
    assert len(names) == 100 and set(names) == set(cmc), 'Release classes differ from CMC'
    synsets = sorted(names)
    by_name = {value: key for key, value in names.items()}
    assert len(by_name) == 100
    size = sum(p['size'] for p in shards)
    free = shutil.disk_usage(root).free
    print(f'{REPO}@{REVISION}: {size / 1e9:.2f} GB download; {free / 1e9:.1f} GB free', flush=True)
    if free < size * 2.5:
        raise RuntimeError('Insufficient disk for shards, images and working space')
    manifest = dict(repo=REPO, revision=REVISION, resized_shorter_side=160,
                    teammate_baseline_verified=False, shards=shards,
                    class_to_idx={s: i for i, s in enumerate(synsets)})
    (root / 'source.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def download(shard):
        path = root / 'downloads' / Path(shard['rfilename']).name
        path.parent.mkdir(exist_ok=True)
        expected = shard['lfs']['sha256']
        if path.exists() and path.stat().st_size == shard['size'] and sha256(path) == expected:
            return path
        temporary = path.with_suffix('.partial')
        # Public download only. Do not read or attach local credentials.
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE + shard['rfilename'], timeout=120) as response:
                    with temporary.open('wb') as output:
                        shutil.copyfileobj(response, output, length=2**20)
                if temporary.stat().st_size != shard['size'] or sha256(temporary) != expected:
                    raise ValueError('Shard size or SHA256 mismatch: ' + path.name)
                temporary.replace(path)
                print('Verified', path.name, flush=True)
                return path
            except Exception:
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(download, shards))
    counts = {'train': Counter(), 'val': Counter()}
    label_mapping = None
    for path in paths:
        parquet = pq.ParquetFile(path)
        metadata = json.loads(parquet.schema_arrow.metadata[b'huggingface'])
        labels = metadata['info']['features']['label']['names']
        mapping = [by_name[name] for name in labels]
        if label_mapping is None:
            label_mapping = mapping
        assert mapping == label_mapping and set(mapping) == set(synsets)
        split = 'val' if path.name.startswith('validation-') else 'train'
        row = 0
        for batch in parquet.iter_batches(batch_size=128, columns=['image', 'label']):
            for image, label in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                synset = mapping[label]
                content = image['bytes']
                with Image.open(io.BytesIO(content)) as picture:
                    picture.verify()
                directory = root / split / synset
                directory.mkdir(parents=True, exist_ok=True)
                destination = directory / f'{path.stem}-{row:06d}.png'
                if not destination.exists():
                    destination.write_bytes(content)
                else:
                    assert hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(content).digest()
                counts[split][synset] += 1
                row += 1
        print('Prepared', path.name, row, 'images', flush=True)
    assert sum(counts['train'].values()) == 126689
    assert sum(counts['val'].values()) == 5000
    assert set(counts['train']) == set(counts['val']) == set(synsets)
    assert all(n == 50 for n in counts['val'].values())
    manifest.update(counts=counts, parquet_label_to_synset=label_mapping)
    (root / 'verified.json').write_text(json.dumps(manifest, indent=2) + '\n')
    config.mkdir(exist_ok=True)
    (config / 'imagenet100_cmc_classes.txt').write_text('\n'.join(synsets) + '\n')
    (config / 'imagenet100_cmc160.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Verified all images, classes and label mappings:', root, flush=True)


if __name__ == '__main__':
    main()
