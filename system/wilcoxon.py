"""
Wilcoxon signed-rank tests of FedSHA (and FedSHA-P) against every method, on the paired per-case Dice of the test
cases. The per-case Dice is saved by `./run.sh test_all <gpu>` to ../results/cases/<run>.json.

Usage (from system/):
    python3 wilcoxon.py
For each method: p-value over all test cases pooled (106 cases), and per client. One-sided test: ours > method.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

CLIENTS = ["LiTS17", "DLDS", "KiTS23", "AMOS22", "BraTS", "BHSD"]
NAMES = {"fedsha": "FedSHA", "ft": "FedSHA-P", "central": "MSA-central", "fedavg": "FedMSA",
         "fedprox": "FedProx", "moon": "MOON", "fedmenu": "Fed-MENU", "fedmosa": "FedMoSA", "local": "MSA-local",
         "ditto": "Ditto", "fedrep": "FedRep", "fedhca2": "FedHCA2", "fedala": "FedALA", "feddbe": "FedDBE",
         "no_window": "w/o window", "no_gate": "w/o gate-informed aggregation",
         "organ_only": "organ region in all blocks"}
GROUPS = [("fedsha", "central,fedavg,fedprox,moon,fedmenu,fedmosa,no_window,no_gate,organ_only"),
          ("ft", "local,ditto,fedrep,fedhca2,fedala,feddbe")]


def load(d, run):
    fn = os.path.join(d, f"{run}.json")
    return json.load(open(fn)) if os.path.exists(fn) else None


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def test(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.allclose(a, b):
        return float("nan")
    return float(wilcoxon(a, b, alternative="greater", zero_method="wilcox").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results/cases")
    a = ap.parse_args()
    for ours, others in GROUPS:
        R = load(a.dir, ours)
        if R is None:
            print(f"missing {ours}.json"); continue
        print(f"\n=== {NAMES[ours]} vs. ===")
        print(f"{'method':26s} {'all cases':>14s}  " + "  ".join(f"{c:>8s}" for c in CLIENTS))
        for m in others.split(","):
            B = load(a.dir, m)
            if B is None:
                print(f"{NAMES.get(m, m):26s} (missing)"); continue
            pa, pb, per = [], [], []
            for k in range(len(CLIENTS)):
                ca, cb = R.get(str(k), {}), B.get(str(k), {})
                cases = sorted(set(ca) & set(cb))
                x, y = [ca[c] for c in cases], [cb[c] for c in cases]
                pa += x; pb += y
                per.append(test(x, y))
            p = test(pa, pb)
            print(f"{NAMES.get(m, m):26s} {p:9.2e} {stars(p):>4s}  " + "  ".join(f"{q:8.3f}" for q in per)
                  + f"   (n = {len(pa)})")


if __name__ == "__main__":
    main()