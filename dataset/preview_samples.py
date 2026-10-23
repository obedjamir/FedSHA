"""
Saves a preview grid of random slices per client: the 2.5D image (centre slice, channel G) with the client's
class (seg_regions merged) as a red overlay, and the plain image next to it.

Usage (from the dataset folder):
    python preview_samples.py                    # modorgan6, 8 slices per client -> previews/<client>.png
    python preview_samples.py --data modorgan6 --n 12 --split test
"""
import argparse
import os
import random

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="modorgan6")
    ap.add_argument("--split", default="train")
    ap.add_argument("--run", default="0")
    ap.add_argument("--n", type=int, default=8, help="slices per client (half with the class, half without)")
    ap.add_argument("--out", default="previews")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(0)
    folder = os.path.join(a.data, f"{a.split}{a.run}")
    for f in sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0])):
        d = np.load(os.path.join(folder, f), allow_pickle=True)["data"].tolist()
        meta = d.get("meta", {}) or {}
        regions = sorted({v for r in meta.get("seg_regions", [[1]]) for v in r})
        pairs = list(zip(d["x"], d["y"]))
        rng.shuffle(pairs)
        pos, neg = [], []
        for img, msk in pairs:                                     # half with the class, half without
            m = np.isin(np.array(Image.open(msk)), regions)
            (pos if m.any() else neg).append((img, m))
            if len(pos) >= a.n // 2 and len(neg) >= a.n - a.n // 2:
                break
        tiles = []
        for img, m in pos[:a.n // 2] + neg[:a.n - a.n // 2]:
            x = np.array(Image.open(img).convert("RGB"))
            g = np.repeat(x[..., 1:2], 3, axis=-1)                 # centre slice as grey
            ov = g.copy()
            ov[m] = (0.5 * ov[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            tiles.append(np.concatenate([g, ov], axis=1))
        h = max(t.shape[0] for t in tiles)
        w = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, h - t.shape[0]), (0, w - t.shape[1]), (0, 0))) for t in tiles]
        grid = np.concatenate([np.concatenate(tiles[i:i + 2], axis=1) if i + 1 < len(tiles)
                               else np.pad(tiles[i], ((0, 0), (0, w), (0, 0))) for i in range(0, len(tiles), 2)], 0)
        name = f"{f.split('.')[0]}_{meta.get('dataset', 'client')}.png"
        Image.fromarray(grid).save(os.path.join(a.out, name))
        print(f"{name}: {len(d['x'])} slices | {meta.get('modality')} {meta.get('organ')} {meta.get('seg_classes')} "
              f"| shown {min(len(pos), a.n // 2)} with class, {min(len(neg), a.n - a.n // 2)} without")


if __name__ == "__main__":
    main()
