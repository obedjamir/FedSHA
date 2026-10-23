"""
Ablation figure (no GPU; reads logs_mo/<run>.log, falls back to the test results below).
(a) Per-client test Dice of FedSHA vs. w/o window. (b) FedSHA vs. w/o gate-informed aggregation.

Usage (from system/):
    python3 ablation_figs.py --out ../results/ablation.pdf
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(log_dir, run):
    fn = os.path.join(log_dir, f"{run}.log")
    if not os.path.exists(fn):
        return None
    L = open(fn, errors="ignore").read().splitlines()
    val = [float(m.group(1)) for l in L for m in [re.search(r"mean_val_score: ([\d.]+)", l)] if m]
    t = [i for i, l in enumerate(L) if "test_loss" in l or "mean_test_score" in l]
    test = None
    if t:
        rows = [l for l in L[t[-1] - 10:t[-1]] if l.startswith("Client ") and "dice_case_client_mean" in l]
        test = [float(re.search(r"dice_case_client_mean: ([\d.]+)", l).group(1)) * 100 for l in rows][-6:]
    return {"val": val, "test": test}


# per-client test Dice, read from logs_mo/<run>.log (test lines at the end); fallback values below
RUNS = {"FedSHA": "fedsha", "w/o window": "no_window", "w/o gate-informed agg.": "no_gate"}
FALLBACK = {"FedSHA": [93.52, 92.24, 90.26, 93.48, 85.66, 45.74],
            "w/o window": [52.44, 87.41, 85.93, 66.56, 69.87, 30.33],
            "w/o gate-informed agg.": [93.20, 92.31, 91.12, 90.09, 85.24, 45.48]}
SHORT = ["LiTS17\n(CT)", "DLDS\n(MRI)", "KiTS23\n(CT)", "AMOS22\n(MRI)", "BraTS\n(MRI)", "BHSD\n(CT)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/ablation.pdf")
    ap.add_argument("--log_dir", default="logs_mo")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.titleweight": "bold", "axes.linewidth": 0.8})
    ABL_DICE = {}
    for k, run in RUNS.items():
        r = read(a.log_dir, run)
        ABL_DICE[k] = r["test"] if r and r["test"] and len(r["test"]) == 6 else FALLBACK[k]
        print(f"{k:24s} ({run}): " + (" ".join(f"{v:.2f}" for v in ABL_DICE[k]) if ABL_DICE[k] else "MISSING"))
    assert ABL_DICE["FedSHA"] is not None, "FedSHA test Dice missing: give logs_mo/fedsha.log or fill FALLBACK"
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    xs = np.arange(6)
    ref = ABL_DICE["FedSHA"]
    for ax, tag, (var, col) in zip(axs.flat[:2], "ab", [("w/o window", "#C9A9A6"), ("w/o gate-informed agg.", "#D8C8A8")]):
        ax.bar(xs - 0.19, ref, 0.36, color="#5B7A99", label="FedSHA", edgecolor="white", linewidth=0.3)
        ax.bar(xs + 0.19, ABL_DICE[var], 0.36, color=col, label=var, edgecolor="white", linewidth=0.3)
        ax.set_xticks(xs); ax.set_xticklabels(SHORT, fontsize=7)
        lo = 0 if var == "w/o window" else 40                      # truncated axis: differences are visible
        ax.set_ylabel("Dice (%)"); ax.set_ylim(lo, 100)
        ax.legend(ncol=2, prop={"weight": "bold", "size": 7}, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.17))
        ax.set_title(f"({tag}) {var}", fontsize=9, y=-0.48)

    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, lw=0.3, alpha=0.6); ax.set_axisbelow(True)

    plt.subplots_adjust(left=0.08, right=0.99, top=0.87, bottom=0.30, wspace=0.22)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out); fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
