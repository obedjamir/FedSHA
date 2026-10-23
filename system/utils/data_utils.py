import os
import random
import hashlib

import numpy as np

DATA_ROOT = '../dataset'

# Run seed (--seed, set by main.py): 32 = the default run; other values give different per-epoch slice draws
# for seed repeats. Train / val / test splits never depend on it.
RUN_SEED = 32


def _stable_seed(*parts):
    """Reproducible across processes (python's hash() is randomised per run)."""
    return int(hashlib.sha256("_".join(map(str, parts)).encode()).hexdigest()[:15], 16)


def _load_npz(dataset, idx, dataset_id, data_split):
    file_path = os.path.join(DATA_ROOT, dataset, f'{data_split}{dataset_id}', f'{idx}.npz')
    with np.load(file_path, allow_pickle=True) as f:
        return f['data'].tolist()


def load_split(dataset, idx, dataset_id, data_split):
    """Slice list of one client: [(image_png, mask_png, organ, cls_label), ...]."""
    d = _load_npz(dataset, idx, dataset_id, data_split)
    cls = d.get('c', [""] * len(d['x']))
    return list(zip(d['x'], d['y'], d['z'], cls))


def read_client_meta(dataset, idx, dataset_id):
    """Label description of one client (modality, organ, seg regions); {} for old npz files."""
    return dict(_load_npz(dataset, idx, dataset_id, 'train').get('meta', {}) or {})


def random_subset(items, dataset, round, idx, dataset_id, epoch=0, max_items=None):
    """A different, reproducible subset of max_items slices every round and local epoch
    (seeded by dataset, round, epoch, client, run)."""
    if len(items) == 0 or not max_items or max_items <= 0:
        return items
    k = min(int(max_items), len(items))
    parts = (dataset, round, epoch, idx, dataset_id) + (() if RUN_SEED == 32 else (RUN_SEED,))
    rng = random.Random(_stable_seed(*parts))
    return rng.sample(items, k)


def read_client_data(dataset, round, idx, dataset_id, data_split='train', epoch=0, max_items=None):
    items = load_split(dataset, idx, dataset_id, data_split)
    if data_split == 'train':
        items = random_subset(items, dataset, round, idx, dataset_id, epoch=epoch, max_items=max_items)
    return items


def _patient_of(img_path):
    """Patient id from the slice path <dataset>/<patient>/<slice>_img.png."""
    p = os.path.normpath(str(img_path)).split(os.sep)
    return p[-2] if len(p) >= 2 else ""


def print_dataset_stats(dataset, num_clients, dataset_id, clients=None):
    """Per client and split: slices and patients; totals at the end."""
    ids = clients if clients else list(range(num_clients))
    print(f"\n=== Dataset statistics: {dataset} (split id {dataset_id}) ===")
    tot = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
    for k in ids:
        meta = read_client_meta(dataset, k, dataset_id)
        head = f"Client {k}" + (f" [{meta.get('modality', '')} {meta.get('organ', '')} | {meta.get('dataset', '')}]"
                                if meta.get("organ") else "")
        seg = meta.get("seg_classes")
        print(head + (f" | seg {seg}" if seg else ""))
        for split in ("train", "val", "test"):
            items = load_split(dataset, k, dataset_id, split)
            pats = {_patient_of(x[0]) for x in items}
            print(f"    {split:5s}: {len(items):6d} slices | {len(pats):5d} patients")
            tot[split][0] += len(items)
            tot[split][1] += len(pats)
    print("Total: " + " | ".join(f"{s} {n} slices / {p} patients" for s, (n, p) in tot.items() if n))
    print("=" * 50, flush=True)
