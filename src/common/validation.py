"""Validate prepared, reviewed datasets without changing their splits."""
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def validate(kind):
    base = ROOT / 'data/processed'
    dataset = base / ('classification' if kind == 'classification' else 'detection_4class')
    manifest = base / 'split_manifest.csv' if kind == 'classification' else dataset / 'manifest.csv'
    rows = list(csv.DictReader(manifest.open()))
    if kind == 'detection':
        config = yaml.safe_load((dataset / 'data.yaml').read_text())
        if config['names'] != dict(enumerate(CLASSES)):
            raise ValueError('Detection class mapping changed')
    expected = {}
    for row in rows:
        if row['split'] not in ('train', 'val', 'test') or row['class_name'] not in CLASSES:
            raise ValueError(f'Invalid manifest label/split: {row}')
        path = (ROOT / row['processed_image'] if kind == 'classification' else
                dataset / 'images' / row['split'] / row['filename'])
        if path in expected:
            raise ValueError(f'Duplicate manifest entry: {path}')
        expected[path] = row
    seen = {}
    counts = Counter()
    version = hashlib.sha256()
    actual = set()
    for split in ('train', 'val', 'test'):
        folders = [dataset / split / name for name in CLASSES] if kind == 'classification' else [dataset / 'images' / split]
        for folder in folders:
            if not folder.is_dir() or not any(folder.iterdir()):
                raise ValueError(f'Missing/empty dataset directory: {folder}')
            if kind == 'classification' and {p.name for p in folder.parent.iterdir() if p.is_dir()} != set(CLASSES):
                raise ValueError('Unexpected classification class directories')
            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    raise ValueError(f'Unexpected nested directory: {path}')
                row = expected.get(path)
                if row is None or row['split'] != split or (kind == 'classification' and row['class_name'] != folder.name):
                    raise ValueError(f'Image has no matching verified manifest label: {path}')
                with Image.open(path) as image:
                    image.load()
                value = digest(path)
                if value != row['hash']:
                    raise ValueError(f'Image differs from reviewed manifest: {path}')
                if value in seen and seen[value] != split:
                    raise ValueError(f'Exact duplicate across splits: {path}')
                seen[value] = split
                actual.add(path)
                version.update(f'{split}:{row["class_name"]}:{value}'.encode())
                counts[f'{split}/{row["class_name"]}'] += 1
                if kind == 'detection':
                    label = dataset / 'labels' / split / f'{path.stem}.txt'
                    lines = label.read_text().splitlines()
                    if not lines or len(lines) != int(row['box_count']):
                        raise ValueError(f'Missing four-class boxes: {label}')
                    for line in lines:
                        parts = line.split()
                        if len(parts) != 5 or parts[0] != str(CLASSES.index(row['class_name'])):
                            raise ValueError(f'Invalid class/YOLO format: {label}')
                        x, y, w, h = map(float, parts[1:])
                        valid_box = (
                            all(math.isfinite(v) for v in (x, y, w, h))
                            and 0 < w <= 1 and 0 < h <= 1
                            and x - w/2 >= -1e-5 and x + w/2 <= 1 + 1e-5
                            and y - h/2 >= -1e-5 and y + h/2 <= 1 + 1e-5
                        )
                        if not valid_box:
                            raise ValueError(f'Invalid normalized box: {label}')
                    version.update(label.read_bytes())
        if kind == 'detection':
            labels = {p.stem for p in (dataset / 'labels' / split).iterdir()}
            if labels != {p.stem for p in actual if p.parent.name == split}:
                raise ValueError(f'Orphan/missing labels in {split}')
    if actual != set(expected):
        raise ValueError('Manifest and dataset file inventory differ')
    result = {'dataset_version': version.hexdigest(), 'valid_images': len(actual), 'class_distribution': dict(counts)}
    print(result)
    return result
