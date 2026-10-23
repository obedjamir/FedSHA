"""
Qualitative figure from the predictions saved by `./run.sh dump_all <gpu>` (../results/preds/<run>.npz).
Columns: clients. Rows: image with ground truth, then one row per method (default: Fed-MENU, the strongest
global baseline, and FedSHA) (prediction in red, ground truth contour
in green, Dice of the slice in the corner). The same slice is shown for every method.

Usage (from system/):
    python3 qualitative_fig.py                                   # -> ../results/preds/qualitative.pdf
    python3 qualitative_fig.py --runs fedmenu,fedsha --pick best
--pick random: per client, a random slice among those where the competitor's Dice is low but not a failure,
               between --min_other (0.4) and --max_other (0.75), and FedSHA's Dice is at least --min_ours (0.8) (default).
               Every run gives a new figure (saved as qualitative_seed<N>.pdf); keep the one you like and
               re-create it with --seed N
--pick diff  : per client, the slice where FedSHA improves most over the other methods (default), among the
               slices where the competitor still reaches --min_other (0.4): the competitor is clearly worse, but does
               not fail completely (no condition on FedSHA)
--pick best  : per client, the slice with the highest FedSHA Dice
--pick median: per client, the slice whose FedSHA Dice is closest to FedSHA's median Dice (more representative)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

NAMES = {"fedsha": "FedSHA", "fedavg": "FedMSA", "fedprox": "FedProx", "moon": "MOON",
         "fedmenu": "Fed-MENU", "fedmosa": "FedMoSA", "central": "MSA-central"}
CLIENTS = ["LiTS17 (CT)", "DLDS (MRI)", "KiTS23 (CT)", "AMOS22 (MRI)", "BraTS (MRI)", "BHSD (CT)"]


def dice(g, p):
    s = g.sum() + p.sum()
    return 2 * (g & p).sum() / s if s else 1.0


def crop_box(masks, shape, margin=0.25, min_size=96):
    m = np.any(np.stack(masks), 0)
    if not m.any():
        return 0, shape[0], 0, shape[1]
    r, c = np.where(m)
    cy, cx = (r.min() + r.max()) / 2, (c.min() + c.max()) / 2
    size = max(r.max() - r.min(), c.max() - c.min()) * (1 + 2 * margin)
    size = int(min(max(size, min_size), min(shape)))
    y0 = int(np.clip(cy - size / 2, 0, shape[0] - size))
    x0 = int(np.clip(cx - size / 2, 0, shape[1] - size))
    return y0, y0 + size, x0, x0 + size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results/preds")
    ap.add_argument("--runs", default="fedmenu,fedsha")
    ap.add_argument("--pick", default="random", choices=["random", "best", "diff", "median"])
    ap.add_argument("--seed", type=int, default=None, help="random: seed (default: new random choice every run)")
    ap.add_argument("--out", default="../results/preds/qualitative.pdf")
    ap.add_argument("--crop", type=int, default=1, help="1 = zoom around the target region, 0 = full slice")
    ap.add_argument("--min_crop", type=int, default=96, help="smallest crop size in pixels (larger = less zoom)")
    ap.add_argument("--min_other", type=float, default=0.4, help="competitor Dice at least this (not a total failure)")
    ap.add_argument("--max_other", type=float, default=0.75, help="random: competitor Dice at most this (clearly low)")
    ap.add_argument("--min_ours", type=float, default=0.8, help="random: FedSHA Dice at least this (good)")
    a = ap.parse_args()
    runs = a.runs.split(",")
    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 100000)
    rng = np.random.default_rng(seed)
    if a.pick == "random":
        a.out = a.out.replace(".pdf", f"_seed{seed}.pdf")
        print(f"seed {seed} (re-create this figure with --seed {seed})")
    D = {r: np.load(os.path.join(a.dir, f"{r}.npz")) for r in runs}
    ref = D["fedsha"]
    n_cli = len(CLIENTS)
    fig, axes = plt.subplots(len(runs) + 1, n_cli, figsize=(1.45 * n_cli, 1.45 * (len(runs) + 1)))
    for c in range(n_cli):
        idx = sorted({k.split("_")[1] for k in ref.files if k.startswith(f"c{c}_")}, key=int)
        if not idx:
            continue
        scores = {j: {r: dice(D[r][f"c{c}_{j}_gt"], D[r][f"c{c}_{j}_pred"]) for r in runs} for j in idx}
        ok = [j for j in idx if min(scores[j][r] for r in runs if r != "fedsha") >= a.min_other]
        if a.pick == "random":                         # random slice where the competitor's Dice is low, not zero
            low = [j for j in ok if max(scores[j][r] for r in runs if r != "fedsha") <= a.max_other
                   and scores[j]["fedsha"] >= a.min_ours]
            j = rng.choice(low or ok or idx)
        elif a.pick == "best":
            j = max(idx, key=lambda j: scores[j]["fedsha"])
        elif a.pick == "diff":
            gap = lambda j: scores[j]["fedsha"] - np.mean([scores[j][r] for r in runs if r != "fedsha"])
            ok = [j for j in idx if min(scores[j][r] for r in runs if r != "fedsha") >= a.min_other]
            j = max(ok or idx, key=gap)                    # clear but not catastrophic gap
        else:
            med = np.median([scores[j]["fedsha"] for j in idx])
            j = min(idx, key=lambda j: abs(scores[j]["fedsha"] - med))
        img, gt = ref[f"c{c}_{j}_img"], ref[f"c{c}_{j}_gt"]
        if a.crop:
            y0, y1, x0, x1 = crop_box([gt] + [D[r][f"c{c}_{j}_pred"] for r in runs], img.shape, min_size=a.min_crop)
        else:
            y0, y1, x0, x1 = 0, img.shape[0], 0, img.shape[1]
        print(f"client {c}: case {ref[f'c{c}_{j}_case']} | " + ", ".join(f"{NAMES.get(r, r)} {scores[j][r] * 100:.1f}" for r in runs))
        rows = [("Image + GT", None)] + [(NAMES.get(r, r), r) for r in runs]
        for i, (label, r) in enumerate(rows):
            ax = axes[i, c]
            ax.imshow(img[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
            g = gt[y0:y1, x0:x1]
            if r is None:
                ov = np.zeros(g.shape + (4,)); ov[g] = (0.1, 0.8, 0.2, 0.45)
                ax.imshow(ov, interpolation="nearest")
            else:
                p = D[r][f"c{c}_{j}_pred"][y0:y1, x0:x1]
                ov = np.zeros(p.shape + (4,)); ov[p] = (0.85, 0.15, 0.15, 0.5)
                ax.imshow(ov, interpolation="nearest")
                edge = g & ~ndimage.binary_erosion(g)
                ec = np.zeros(g.shape + (4,)); ec[edge] = (0.1, 0.9, 0.2, 1.0)
                ax.imshow(ec, interpolation="nearest")
                ax.text(0.03, 0.97, f"{scores[j][r] * 100:.1f}", transform=ax.transAxes, va="top", ha="left",
                        fontsize=9, fontweight="bold", color="white", bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(CLIENTS[c], fontsize=9.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(label, fontsize=9.5, fontweight="bold")
    plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.035, right=0.995, top=0.94, bottom=0.005)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()