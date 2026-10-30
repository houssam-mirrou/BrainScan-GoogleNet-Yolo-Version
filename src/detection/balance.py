"""Reproduce notebook 04's seeded four-class training resampling."""
from pathlib import Path
import shutil

import pandas as pd
import yaml

from src.common.validation import CLASSES

SEED = 42


def prepare_balanced_dataset(source, destination):
    """Balance training images; copy validation/test images and labels unchanged."""
    source, destination = Path(source), Path(destination)
    frame = pd.read_csv(source / 'manifest.csv', dtype={'hash': str})
    train = frame[frame['split'] == 'train']
    counts = train['class_name'].value_counts()
    if any(counts.get(name, 0) == 0 for name in CLASSES):
        raise ValueError('Every class needs training images before balancing')
    target = min(counts[name] for name in CLASSES if name != 'No Tumor')
    destination.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val', 'test'):
        for folder in ('images', 'labels'):
            (destination / folder / split).mkdir(parents=True)

    records = []

    def copy_sample(row, filename):
        split = row['split']
        shutil.copy2(source / 'images' / split / row['filename'],
                     destination / 'images' / split / filename)
        shutil.copy2(source / 'labels' / split / f"{Path(row['filename']).stem}.txt",
                     destination / 'labels' / split / f'{Path(filename).stem}.txt')
        records.append({**row, 'filename': filename})

    for class_id, name in enumerate(CLASSES):
        group = train[train['class_name'] == name]
        if len(group) >= target:
            selected = group.sample(n=target, replace=False, random_state=SEED)
        else:
            extra = group.sample(n=target - len(group), replace=True, random_state=SEED)
            selected = pd.concat([group, extra], ignore_index=True)
        for index, row in enumerate(selected.to_dict('records')):
            filename = f"{class_id}_{index:05d}_{row['hash']}{Path(row['filename']).suffix}"
            copy_sample(row, filename)

    for row in frame[frame['split'].isin(['val', 'test'])].to_dict('records'):
        copy_sample(row, row['filename'])

    pd.DataFrame(records).to_csv(destination / 'manifest.csv', index=False)
    (destination / 'data.yaml').write_text(yaml.safe_dump(dict(
        path=str(destination.resolve()), train='images/train', val='images/val',
        test='images/test', names=dict(enumerate(CLASSES)),
    )))
    return dict(seed=SEED, target_per_class=int(target),
                original_train_counts={name: int(counts[name]) for name in CLASSES})
