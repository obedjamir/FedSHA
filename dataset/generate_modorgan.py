"""
Builds the modality x organ federation (segmentation only): one dataset per client, ONE class per client.

    client  dataset    modality  organ    class (mask labels merged)          series used as a "volume"
    0       LiTS17     CT        liver    liver (1+2)                         the CT volume
    1       DLDS       MR        liver    liver                               every labelled series (sequence)
    2       KiTS23     CT        kidney   kidney (1+2+3)                      the CT volume (from the 2D export)
    3       AMOS22     MR        kidney   kidney (right 2 + left 3)           every labelled MR volume (id > 500)
    4       BraTS      MR        brain    whole tumor (1+2+4)                 FLAIR
    5       BHSD       CT        brain    hemorrhage (1-5, all subtypes)      the head CT volume
    Every organ is held by one CT and one MR client. LIDC-IDRI (CT lung nodule) and CHAOS are on standby: their
    builders are kept, add them to CLIENTS to use them.

The abdominal clients segment the whole organ (tumors are part of the organ, no tumor class); BraTS and BHSD
segment the affected region (tumor / hemorrhage).

Every slice is stored as a PNG pair:
    image : RGB uint8, 2.5D = slices z-1, z, z+1 of ONE series (clamped at the ends).
            CT: fixed organ HU window. MR: percentile window [0.5, 99.5] of the whole series
            (BraTS: over brain voxels), so neighbouring and empty slices keep their relative brightness.
    mask  : uint8 label map with the dataset's own labels; the client's seg_regions merge them into one class.
Slices: every slice with the class (organ or affected region) + an equal number of slices without it, same series.
Split: 80 / 10 / 10 by PATIENT (all series of a patient stay in one split).
npz: data={"x": image paths, "y": mask paths, "z": organ, "c": "", "meta": {organ, modality, seg_classes, seg_regions}}.

Requirements: nibabel, pydicom, pylidc (LIDC), opencv-python.
Nothing existing is deleted: the npz files go to a new folder (dir_path), the PNG slices to a new slice folder
(slice_root), and a client whose npz files already exist is skipped (rebuild it with --only <name> --force).

Run from the dataset folder (next to utils/):
    python generate_modorgan.py                      # every client that is not built yet
    python generate_modorgan.py --only amos,bhsd     # some clients
    python generate_modorgan.py --only kits --force  # rebuild one client (overwrites only that client)
"""
import argparse
import os
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import sklearn

from utils.dataset_utils import (save_pair, choose_slices, write_volume, middle_slices, save_split,
                                 stack_25d, percentile_window, TRAIN_FRAC)

# pylidc still uses aliases removed in numpy >= 1.24
for _alias, _type in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sklearn.utils.check_random_state(seed)


runs = 1
dir_path = "modorgan6/"
DATA_ROOT = "/DATA1/obed/jamir/data"
slice_root = f"{DATA_ROOT}/MO6_slices"                             # PNG slices (new folder: old slices are kept)

# ------------------------------------------------------------------
# Dataset roots
# ------------------------------------------------------------------
LITS_ROOT = f"{DATA_ROOT}/LiTS17/data"                             # volume-*.nii / segmentation-*.nii (any depth)
KITS_ROOT = f"{DATA_ROOT}/KITS23_2D/KiTS23_2D"                     # images/ + masks/ : {case}_slice_{zzz}.png
KITS_WINDOW = (-200, 300)                                          # HU window of the 16-bit 2D export
LIDC_ROOT = f"{DATA_ROOT}/LIDC-IDRI"                               # one folder per SeriesInstanceUID with the .dcm files
DLDS_ROOT = f"{DATA_ROOT}/DLDS/Segmentation"                       # <patient>/<series>/{images,masks}
CHAOS_ROOT = f"{DATA_ROOT}/CHAOS/Train_Sets/MR"                    # <patient>/{T1DUAL,T2SPIR}/...
AMOS_ROOT = f"{DATA_ROOT}/amos22"                                # imagesTr/Va/Ts + labelsTr/Va/Ts (nnU-Net layout)
BHSD_ROOT = f"{DATA_ROOT}/label_192"                              # images/<id>.nii.gz + <label folder>/<id>.nii.gz
BRATS_ROOT = f"{DATA_ROOT}/BraTS/GBM_split"                        # {train,val,...}/<subject>/*_flair.nii.gz + *_GlistrBoost_ManuallyCorrected.nii.gz

# DLDS series with liver masks (as in FedMoSA)
DLDS_SERIES = {
    '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
    '16', '17', '18', '19', '20', '21', '22', '23', '25', '26', '27', '29',
    '30', '31', '32', '34', '36', '38', '43', '60', '503', '504', '603',
    '604', '705', '706', '803', '804', '904', '2205', '2206'
}
DLDS_MAX_SERIES_PER_PATIENT = None      # None = every labelled series; set e.g. 2 if DLDS dominates the federation
# CHAOS ground truth (PNG grey values): liver 55-70, right kidney 110-135, left kidney 175-200, spleen 240-255
CHAOS_KIDNEY = [(110, 135), (175, 200)]
AMOS_KIDNEY = (2, 3)                    # AMOS labels: 2 right kidney, 3 left kidney
AMOS_MR_MIN_ID = 500                    # AMOS ids > 500 are MR
BRATS_SEQUENCE = "flair"
BRATS_MIN_BRAIN = 0.25                  # tumor-free negatives: slices with >= 25% of the largest brain area

# ------------------------------------------------------------------
# Federation (client id = position in this list)
#   seg_classes / seg_regions: one output channel (slot) per region; a region is a set of mask labels
# ------------------------------------------------------------------
CLIENTS = [
    {"name": "lits",  "organ": "liver",  "modality": "CT",
     "meta": {"seg_classes": ["liver"], "seg_regions": [[1, 2]]}},
    {"name": "dlds",  "organ": "liver",  "modality": "MR",
     "meta": {"seg_classes": ["liver"], "seg_regions": [[1]]}},
    {"name": "kits",  "organ": "kidney", "modality": "CT",
     "meta": {"seg_classes": ["kidney"], "seg_regions": [[1, 2, 3]]}},
    {"name": "amos",  "organ": "kidney", "modality": "MR",
     "meta": {"seg_classes": ["kidney"], "seg_regions": [[1]]}},
    {"name": "brats", "organ": "brain",  "modality": "MR",
     "meta": {"seg_classes": ["tumor"], "seg_regions": [[1, 2, 4]]}},
    {"name": "bhsd",  "organ": "brain",  "modality": "CT",
     "meta": {"seg_classes": ["hemorrhage"], "seg_regions": [[1, 2, 3, 4, 5]]}},
]
STANDBY = {                                                         # not in the federation (add to CLIENTS to use)
    "lidc":  {"name": "lidc",  "organ": "lung",   "modality": "CT",
              "meta": {"seg_classes": ["nodule"], "seg_regions": [[1]]}},
    "chaos": {"name": "chaos", "organ": "kidney", "modality": "MR",
              "meta": {"seg_classes": ["kidney"], "seg_regions": [[1]]}},
}


# ==================================================================
# CT builders
# ==================================================================
def build_lits(c, out_dir, rng):
    """Liver slices (labels 1 + 2) + as many liver-free slices. Reads only the chosen CT slices (.nii is seekable)."""
    import nibabel as nib
    vols = {p.name.split("-")[1].split(".")[0]: p for p in Path(LITS_ROOT).rglob("volume-*.nii*")}
    segs = {p.name.split("-")[1].split(".")[0]: p for p in Path(LITS_ROOT).rglob("segmentation-*.nii*")}
    ids = sorted(set(vols) & set(segs))
    rng.shuffle(ids)
    patients = defaultdict(list)
    for vid in ids:
        seg = np.asarray(nib.load(str(segs[vid])).dataobj).astype(np.uint8)
        organ = (seg >= 1).any(axis=(0, 1))                        # axial axis = 2
        pos = np.where(organ)[0]
        if len(pos) == 0:
            continue
        neg_cand = np.where(~organ)[0]
        img = nib.load(str(vols[vid])).dataobj
        if str(vols[vid]).endswith(".gz"):                         # gzip cannot seek: read once
            img = np.asarray(img, dtype=np.float32)
        pid = f"lits{vid}"
        get = lambda k, img=img: np.asarray(img[:, :, int(k)], dtype=np.float32)
        for z in choose_slices(pos, neg_cand, rng):
            x = stack_25d(get, int(z), seg.shape[2], organ=c["organ"])
            patients[pid].append(save_pair(x, seg[:, :, z], out_dir / pid, f"{pid}_{z:04d}") + ("",))
    return patients


def build_kits(c, out_dir, rng):
    """KiTS23 from the 2D export (16-bit PNG windowed to KITS_WINDOW; masks 1 kidney, 2 tumor, 3 cyst).
    Kidney slices (labels 1-3) + as many kidney-free slices; images mapped back to HU and re-windowed."""
    import cv2
    root = Path(KITS_ROOT)
    by_case = defaultdict(list)
    for p in (root / "masks").glob("case_*_slice_*.png"):
        case, z = p.stem.rsplit("_slice_", 1)
        by_case[case].append((int(z), p.name))
    cases = sorted(by_case)
    rng.shuffle(cases)
    lo, hi = KITS_WINDOW
    patients = defaultdict(list)
    for case in cases:
        slices = sorted(by_case[case])
        labels = {z: cv2.imread(str(root / "masks" / f), cv2.IMREAD_UNCHANGED) for z, f in slices}
        pos = [z for z, m in labels.items() if (m >= 1).any()]
        if not pos:
            continue
        neg_cand = [z for z, m in labels.items() if not (m >= 1).any()]

        def get(k, case=case):
            p = root / "images" / f"{case}_slice_{k:03d}.png"
            if not p.exists():
                return None
            return lo + cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0 * (hi - lo)

        n = max(z for z, _ in slices) + 1
        for z in choose_slices(pos, neg_cand, rng):
            x = stack_25d(get, int(z), n, organ=c["organ"])
            patients[case].append(save_pair(x, labels[z].astype(np.uint8), out_dir / case, f"{case}_{z:04d}") + ("",))
    return patients


def _pylidc():
    """pylidc with DICOMs read from <LIDC_ROOT>/<SeriesInstanceUID>."""
    import pylidc as pl
    pl.Scan.get_path_to_dicom_files = lambda scan: str(Path(LIDC_ROOT) / scan.series_instance_uid)
    return pl


def build_lidc(c, out_dir, rng):
    """Mask = 50% consensus of every nodule with >= 2 readers (label 1). Nodule slices + as many nodule-free
    slices from the middle of the scan."""
    from pylidc.utils import consensus
    pl = _pylidc()
    scans = [s for s in pl.query(pl.Scan).all() if (Path(LIDC_ROOT) / s.series_instance_uid).is_dir()]
    print(f"LIDC: {len(scans)} annotated scans found on disk")
    rng.shuffle(scans)
    patients = defaultdict(list)
    for scan in scans:
        nodules = [a for a in scan.cluster_annotations() if len(a) >= 2]
        if not nodules:
            continue
        vol = scan.to_volume(verbose=False)                        # (H, W, Z), HU; axial axis = 2
        lesion = np.zeros(vol.shape, dtype=bool)
        for anns in nodules:
            cmask, cbbox, _ = consensus(anns, clevel=0.5)
            lesion[cbbox] |= cmask
        pos = np.where(lesion.any(axis=(0, 1)))[0]
        if len(pos) == 0:
            continue
        neg_cand = [z for z in middle_slices(vol.shape[2]) if z not in set(pos)]
        pid = scan.patient_id
        patients[pid] += write_volume(vol, lesion, 2, pid, out_dir, choose_slices(pos, neg_cand, rng),
                                      organ=c["organ"], stem=f"{pid}_{scan.series_instance_uid[-6:]}")
    return patients


# ==================================================================
# MR builders
# ==================================================================
def _read_2d(path):
    """One slice file: DICOM (pixel array + z position) or an image file (z unknown)."""
    p = str(path)
    if p.lower().endswith((".dcm", ".dicom")) or "." not in os.path.basename(p):
        import pydicom
        d = pydicom.dcmread(p)
        z = None
        if hasattr(d, "ImagePositionPatient"):
            z = float(d.ImagePositionPatient[2])
        elif hasattr(d, "InstanceNumber"):
            z = float(d.InstanceNumber)
        return d.pixel_array.astype(np.float32), z
    import cv2
    return cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32), None


def _files(d):
    return sorted(f for f in Path(d).iterdir() if f.is_file() and not f.name.startswith("."))


def read_series(img_dir, msk_dir):
    """A 2D-file series -> (vol (n, H, W), masks (n, H, W)) in through-plane order.
    Images and masks are paired by sorted file name (as in FedMoSA); slices are then ordered by the DICOM
    position of the images, so z-1 / z+1 are true neighbours. None if the series cannot be paired."""
    imgs, msks = _files(img_dir), _files(msk_dir)
    if len(imgs) != len(msks) or not imgs:
        print(f"  skip {img_dir}: {len(imgs)} images vs {len(msks)} masks")
        return None
    data = [(_read_2d(i), _read_2d(m)[0]) for i, m in zip(imgs, msks)]
    shape = data[0][0][0].shape
    if any(im.shape != shape or mk.shape != shape for (im, _), mk in data):
        print(f"  skip {img_dir}: slice shapes differ")
        return None
    zs = [z for (_, z), _ in data]
    order = sorted(range(len(data)), key=lambda i: zs[i]) if all(z is not None for z in zs) else range(len(data))
    vol = np.stack([data[i][0][0] for i in order])
    masks = np.stack([data[i][1] for i in order])
    return vol, masks


def _write_series(vol, labels, pid, tag, out_dir, rng, neg_ok=None):
    """Foreground slices of one MR series + as many foreground-free slices, 2.5D in the series' percentile window."""
    fg = labels.reshape(len(labels), -1).any(1)
    pos = np.where(fg)[0]
    if len(pos) == 0:
        return []
    neg_cand = [z for z in range(len(labels)) if not fg[z] and (neg_ok is None or neg_ok[z])]
    window = percentile_window(vol)
    return write_volume(vol, labels, 0, pid, out_dir, choose_slices(pos, neg_cand, rng), window=window,
                        stem=f"{pid}_{tag}")


def build_dlds(c, out_dir, rng):
    """Every labelled series of every patient (several sequences per patient); mask > 0 = liver."""
    root = Path(DLDS_ROOT)
    patients = defaultdict(list)
    pids = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit())
    rng.shuffle(pids)
    n_series = 0
    for pid in pids:
        series = sorted(d for d in (root / pid).iterdir()
                        if d.is_dir() and d.name in DLDS_SERIES and (d / "images").is_dir() and (d / "masks").is_dir())
        rng.shuffle(series)
        if DLDS_MAX_SERIES_PER_PATIENT:
            series = series[:DLDS_MAX_SERIES_PER_PATIENT]
        for s in series:
            r = read_series(s / "images", s / "masks")
            if r is None:
                continue
            vol, m = r
            items = _write_series(vol, (m > 0).astype(np.uint8), f"dlds{pid}", f"s{s.name}", out_dir, rng)
            patients[f"dlds{pid}"] += items
            n_series += bool(items)
    print(f"DLDS: {n_series} series from {len(patients)} patients")
    return patients


def build_chaos(c, out_dir, rng):
    """Kidneys (right + left -> label 1) in T1-InPhase, T1-OutPhase (both with the T1DUAL ground truth) and T2-SPIR."""
    root = Path(CHAOS_ROOT)
    patients = defaultdict(list)
    pids = sorted(d.name for d in root.iterdir() if d.is_dir())
    rng.shuffle(pids)
    for pid in pids:
        p = root / pid
        for tag, img_dir, gt_dir in (("t1in", p / "T1DUAL" / "DICOM_anon" / "InPhase", p / "T1DUAL" / "Ground"),
                                     ("t1out", p / "T1DUAL" / "DICOM_anon" / "OutPhase", p / "T1DUAL" / "Ground"),
                                     ("t2", p / "T2SPIR" / "DICOM_anon", p / "T2SPIR" / "Ground")):
            if not (img_dir.is_dir() and gt_dir.is_dir()):
                continue
            r = read_series(img_dir, gt_dir)
            if r is None:
                continue
            vol, gt = r
            kidney = np.zeros(gt.shape, dtype=np.uint8)
            for lo, hi in CHAOS_KIDNEY:
                kidney[(gt >= lo) & (gt <= hi)] = 1
            patients[f"chaos{pid}"] += _write_series(vol, kidney, f"chaos{pid}", tag, out_dir, rng)
    return patients


def build_brats(c, out_dir, rng):
    """FLAIR only; whole tumor (labels 1, 2, 4 kept as they are, the slot merges them). Window over brain voxels.
    Negatives: tumor-free slices with at least BRATS_MIN_BRAIN of the largest brain area."""
    import nibabel as nib
    subjects = sorted(d for split in Path(BRATS_ROOT).iterdir() if split.is_dir()
                      for d in split.iterdir() if d.is_dir())
    rng.shuffle(subjects)
    patients = defaultdict(list)
    for d in subjects:
        img = list(d.glob(f"*_{BRATS_SEQUENCE}.nii*"))
        seg = list(d.glob("*_GlistrBoost_ManuallyCorrected.nii*"))
        if not img or not seg:
            continue
        vol = np.asarray(nib.load(str(img[0])).dataobj, dtype=np.float32)
        lab = np.asarray(nib.load(str(seg[0])).dataobj).astype(np.uint8)
        brain = vol > 0
        area = brain.sum(axis=(0, 1))
        tumor = (lab > 0).any(axis=(0, 1))
        pos = np.where(tumor)[0]
        if len(pos) == 0:
            continue
        neg_cand = [z for z in range(vol.shape[2]) if not tumor[z] and area[z] >= BRATS_MIN_BRAIN * area.max()]
        pid = d.name
        patients[pid] += write_volume(vol, lab, 2, pid, out_dir, choose_slices(pos, neg_cand, rng),
                                      window=percentile_window(vol, brain), stem=f"{pid}_{BRATS_SEQUENCE}")
    return patients


def _canonical(path):
    """NIfTI in RAS+ (closest canonical), so axis 2 is the axial (inferior -> superior) axis."""
    import nibabel as nib
    return nib.as_closest_canonical(nib.load(str(path)))


def build_amos(c, out_dir, rng):
    """AMOS22 MR volumes (id > AMOS_MR_MIN_ID) from every labelled split; right + left kidney -> label 1.
    Kidney slices + as many kidney-free slices from the middle of the volume; per-volume percentile window."""
    root = Path(AMOS_ROOT)
    pairs = {}
    for split in ("Tr", "Va", "Ts"):
        for lab in sorted((root / f"labels{split}").glob("amos_*.nii*")):
            img = root / f"images{split}" / lab.name
            if img.exists() and int(lab.name.split("_")[1].split(".")[0]) > AMOS_MR_MIN_ID:
                pairs[lab.name.split(".")[0]] = (img, lab)
    print(f"AMOS: {len(pairs)} labelled MR volumes")
    ids = sorted(pairs)
    rng.shuffle(ids)
    patients = defaultdict(list)
    for pid in ids:
        img, lab = pairs[pid]
        vol = np.asarray(_canonical(img).dataobj, dtype=np.float32)
        seg = np.asarray(_canonical(lab).dataobj).astype(np.uint8)
        if vol.shape != seg.shape:
            print(f"  skip {pid}: image {vol.shape} vs label {seg.shape}")
            continue
        kidney = np.isin(seg, AMOS_KIDNEY).astype(np.uint8)
        pos = np.where(kidney.any(axis=(0, 1)))[0]
        if len(pos) == 0:
            continue
        neg_cand = [z for z in middle_slices(vol.shape[2]) if z not in set(pos)]
        patients[pid] += write_volume(vol, kidney, 2, pid, out_dir, choose_slices(pos, neg_cand, rng),
                                      window=percentile_window(vol), stem=f"{pid}_mr")
    return patients


def _bhsd_label_dir(root):
    """The label folder next to images/ (BHSD names it e.g. 'ground truths')."""
    cands = [d for d in root.iterdir() if d.is_dir() and d.name != "images" and not d.name.startswith((".", "__"))]
    assert len(cands) == 1, f"BHSD: expected one label folder next to images/, found {[d.name for d in cands]}"
    return cands[0]


def build_bhsd(c, out_dir, rng):
    """BHSD head CT; every hemorrhage subtype (labels 1-5) kept, the slot merges them. Brain window (0, 80) HU.
    Negatives: hemorrhage-free slices with at least BRATS_MIN_BRAIN of the largest brain (0-80 HU) area."""
    root = Path(BHSD_ROOT)
    lab_dir = _bhsd_label_dir(root)
    ids = sorted(p.name for p in (root / "images").glob("*.nii*") if (lab_dir / p.name).exists())
    print(f"BHSD: {len(ids)} labelled scans (labels in '{lab_dir.name}/')")
    rng.shuffle(ids)
    patients = defaultdict(list)
    for name in ids:
        vol = np.asarray(_canonical(root / "images" / name).dataobj, dtype=np.float32)
        lab = np.asarray(_canonical(lab_dir / name).dataobj).astype(np.uint8)
        if vol.shape != lab.shape:
            print(f"  skip {name}: image {vol.shape} vs label {lab.shape}")
            continue
        bleed = (lab > 0).any(axis=(0, 1))
        pos = np.where(bleed)[0]
        if len(pos) == 0:
            continue
        area = ((vol >= 0) & (vol <= 80)).sum(axis=(0, 1))
        neg_cand = [z for z in range(vol.shape[2]) if not bleed[z] and area[z] >= BRATS_MIN_BRAIN * area.max()]
        pid = "bhsd_" + name.split(".")[0]
        patients[pid] += write_volume(vol, lab, 2, pid, out_dir, choose_slices(pos, neg_cand, rng),
                                      organ=c["organ"], stem=pid)
    return patients


BUILDERS = {"lits": build_lits, "dlds": build_dlds, "kits": build_kits,
            "chaos": build_chaos, "lidc": build_lidc, "brats": build_brats,
            "amos": build_amos, "bhsd": build_bhsd}


# ------------------------------------------------------------------
def generate_modorgan(id, seed, only=None, force=False):
    paths = [dir_path + split + id + "/" for split in ("train", "val", "test")]
    for p in paths:
        os.makedirs(p, exist_ok=True)                              # never deletes an existing federation
    train_path, val_path, test_path = paths

    rng = random.Random(seed)
    print("\n=== Client Data Distribution Summary ===")
    for k, c in enumerate(CLIENTS):
        if only and c["name"] not in only:
            continue
        done = all(os.path.exists(os.path.join(p, f"{k}.npz")) for p in paths)
        if done and not force:
            print(f"Client {k} [{c['name']}]: already built in {dir_path}, skipped (use --force to rebuild)")
            continue
        build_client(k, c, random.Random(seed * 1000 + k), train_path, val_path, test_path)
    print(f"--num_clients {len(CLIENTS)}")


def build_client(k, c, rng, train_path, val_path, test_path):
    """Builds the slices of one client, splits them 80 / 10 / 10 by patient and saves the npz files."""
    patients = BUILDERS[c["name"]](c, Path(slice_root) / c["name"], rng)
    pids = sorted(patients)
    rng.shuffle(pids)
    n_train = int(TRAIN_FRAC * len(pids))
    n_test = (len(pids) - n_train) // 2
    groups = (pids[:n_train], pids[n_train:n_train + n_test], pids[n_train + n_test:])
    train, test, val = [[s for p in g for s in patients[p]] for g in groups]
    meta = dict(c["meta"], organ=c["organ"], modality=c["modality"], dataset=c["name"])
    for path, items in ((train_path, train), (test_path, test), (val_path, val)):
        save_split(path, k, items, c["organ"], meta)
    print(f"Client {k} [{c['name']} | {c['modality']} {c['organ']} | {c['meta']['seg_classes']}]: "
          f"patients {len(groups[0])}/{len(groups[2])}/{len(groups[1])} (train/val/test) | "
          f"slices {len(train)}/{len(val)}/{len(test)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated client names (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild clients that already exist")
    a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",") if x.strip()} or None
    for i in range(runs):
        set_seed(i)
        generate_modorgan(str(i), i, only=only, force=a.force)