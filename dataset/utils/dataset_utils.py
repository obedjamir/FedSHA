import os
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ------------------------------------------------------------------
# Slice settings shared by all clients
# ------------------------------------------------------------------
WINDOWS = {                      # (low, high) HU per organ (CT)
    "liver": (-160, 240),
    "kidney": (-160, 240),
    "lung": (-1000, 400),
    "brain": (0, 80),            # BHSD head CT (brain window)
}
MR_PERCENTILES = (0.5, 99.5)     # MR: per-series percentile window (computed over the whole series)
CONTEXT = 1                      # 2.5D: neighbouring slices on each side (R = z-1, G = z, B = z+1)
MAX_POS_PER_VOLUME = None        # foreground slices kept per volume (None = all)
NEG_PER_POS = 1.0                # foreground-free slices per foreground slice
TRAIN_FRAC = 0.8                 # patient-level 80 / 10 / 10


def check_alt(config_path, train_path, val_path, test_path, num_clients, num_classes):
    """Deletes an existing federation (config + split folders) and recreates the folders."""
    if os.path.exists(config_path):
        print(f"Deleting existing dataset configuration at {config_path}...")
        os.remove(config_path)
    for path in [train_path, val_path, test_path]:
        dir_path = os.path.dirname(path)
        if os.path.exists(dir_path):
            print(f"Deleting existing directory {dir_path}...")
            shutil.rmtree(dir_path)
        print(f"Creating new directory {dir_path}...")
        os.makedirs(dir_path)
    return False


# ------------------------------------------------------------------
# Intensity windows
# ------------------------------------------------------------------
def window_to_uint8(x, organ=None, window=None):
    lo, hi = window if window is not None else WINDOWS[organ]
    x = np.clip(x.astype(np.float32), lo, hi)
    return ((x - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)


def percentile_window(vol, mask=None, pct=MR_PERCENTILES):
    """MR: (low, high) intensity window of a whole series (optionally over `mask` voxels only, e.g. the brain).
    One window per series, so neighbouring slices and empty slices keep their relative brightness."""
    v = np.asarray(vol, dtype=np.float32)
    v = v[mask] if mask is not None and mask.any() else v.reshape(-1)
    lo, hi = np.percentile(v, pct)
    return float(lo), float(max(hi, lo + 1e-3))


def stack_25d(get_slice, z, n, organ=None, window=None):
    """2.5D input: slices z-1, z, z+1 of ONE series (clamped at the ends) as RGB (H, W, 3).
    get_slice(k) -> raw slice k (HU for CT, scanner units for MR; None if missing: the centre slice is used).
    Window: the organ's HU window (CT) or the series' percentile window (MR). Labels stay those of slice z."""
    centre = get_slice(z)
    chans = []
    for d in (-CONTEXT, 0, CONTEXT):
        k = min(max(z + d, 0), n - 1)
        s = centre if k == z else get_slice(k)
        if s is None or s.shape != centre.shape:
            s = centre
        chans.append(window_to_uint8(s, organ, window))
    return np.stack(chans, axis=-1)


# ------------------------------------------------------------------
# Slice writing
# ------------------------------------------------------------------
def save_pair(img_u8, mask, out_dir, stem):
    """Writes an image / label-map PNG pair and returns their paths.
    The mask keeps its label values (0 = background, 1, 2, ... = classes); booleans become 0/1."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{stem}_img.png"
    msk_path = out_dir / f"{stem}_msk.png"
    Image.fromarray(img_u8).save(img_path)
    Image.fromarray(np.asarray(mask).astype(np.uint8)).save(msk_path)
    return str(img_path), str(msk_path)


def choose_slices(pos_idx, neg_candidates, rng):
    """All foreground slices (or up to MAX_POS_PER_VOLUME) + NEG_PER_POS times as many foreground-free slices."""
    pos_idx = list(pos_idx)
    rng.shuffle(pos_idx)
    if MAX_POS_PER_VOLUME is not None:
        pos_idx = pos_idx[:MAX_POS_PER_VOLUME]
    taken = set(pos_idx)
    neg = [i for i in neg_candidates if i not in taken]
    rng.shuffle(neg)
    neg = neg[:int(round(len(pos_idx) * NEG_PER_POS))]
    return sorted(pos_idx + neg)


def write_volume(vol, labels, axis, pid, out_dir, slices, organ=None, window=None, stem=None):
    """Writes the chosen slices of one series as 2.5D images; returns [(img_path, msk_path, "")]."""
    items = []
    n = vol.shape[axis]
    get = lambda k: np.take(vol, k, axis=axis)
    stem = stem or pid
    for z in slices:
        msk = np.take(labels, z, axis=axis)
        img = stack_25d(get, int(z), n, organ, window)
        items.append(save_pair(img, msk, Path(out_dir) / pid, f"{stem}_{z:04d}") + ("",))
    return items


def middle_slices(n, lo=0.2, hi=0.8):
    """Slice indices in the central part of a scan (negatives where no organ mask exists)."""
    return list(range(int(n * lo), int(n * hi)))


# ------------------------------------------------------------------
# One client = one dataset: split by patient and save
# ------------------------------------------------------------------
def split_by_patient(patients, rng, train_frac=TRAIN_FRAC):
    """patients: {patient_id: [(img, msk, cls), ...]} -> train, test, val slice lists (patient-disjoint)."""
    pids = sorted(patients)
    rng.shuffle(pids)
    n_train = int(train_frac * len(pids))
    n_test = (len(pids) - n_train) // 2
    groups = (pids[:n_train], pids[n_train:n_train + n_test], pids[n_train + n_test:])
    return [[pair for p in g for pair in patients[p]] for g in groups]


def save_split(path, client_id, items, organ, meta=None):
    """npz with data={"x": image paths, "y": mask paths, "z": organ names, "c": "" (no image labels),
    "meta": label / modality description}."""
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, f"{client_id}.npz"),
             data={"x": [p[0] for p in items], "y": [p[1] for p in items], "z": [organ for _ in items],
                   "c": [p[2] if len(p) > 2 else "" for p in items], "meta": meta or {}})