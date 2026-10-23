"""
Communication cost per method: trainable parameters and parameters uploaded by one client per round
(download is the same size). No checkpoint / data needed (MedSAM weights are not loaded).

Usage (from system/):  python3 comm_cost.py
"""
import torch
from flcore.trainmodel import models as Mo
from flcore.trainmodel.models import MSAModel, MoSAModel, MENUModel, FedSHA, param_group

# clients of modorgan6: (name, organ index used by FedSHA / Fed-MENU, modality)
CLIENTS = [("LiTS17", "liver", "CT"), ("DLDS", "liver", "MR"), ("KiTS23", "kidney", "CT"),
           ("AMOS22", "kidney", "MR"), ("BraTS", "brain", "MR"), ("BHSD", "brain", "CT")]
ORGANS = ["brain", "kidney", "liver"]            # FedSHA grid regions: brain -> 0, kidney -> 1, liver -> 2
M = 1e6


def groups(model):
    g = {}
    for n, p in model.named_parameters():
        if any(f"output_hypernetworks_mlps.{k}." in n for k in (1, 2, 3)):
            continue                                # unused output heads (one class per client): never updated
        if p.requires_grad:
            k, sub, idx = param_group(n)
            g.setdefault((k, sub, idx), 0)
            g[(k, sub, idx)] += p.numel()
    return g


def total(g, keep=lambda k: True):
    return sum(v for k, v in g.items() if keep(k))


def main():
    Mo.SHARED_HEAD = True
    rows = []
    with torch.no_grad():
        # MSA-based methods (FedMSA / FedProx / MOON / Ditto / SCAFFOLD / FedRep)
        g = groups(MSAModel(checkpoint_path=None))
        tr = total(g)
        head = total(g, lambda k: k[0] in ("slot", "tokens") or False)
        rows += [("FedMSA / FedProx / MOON / Ditto", tr, tr),
                 ("SCAFFOLD (+ control variates)", tr, 2 * tr)]
        Mo.SHARED_HEAD = False
        gh = groups(MSAModel(checkpoint_path=None))
        head = total(gh, lambda k: k[0] in ("slot", "tokens"))
        rows += [("FedRep / FedHCA2 (head local)", total(gh), total(gh) - head)]
        Mo.SHARED_HEAD = True

        # FedMoSA: own-modality specialised adapters + shared adapters + decoder (+ prototypes, negligible)
        g = groups(MoSAModel(checkpoint_path=None))
        up = [total(g, lambda k, m=m: k[0] in ("shared", "decoder") or (k[0] == "spec" and k[1] == m))
              for _, _, m in CLIENTS]
        rows.append(("FedMoSA", total(g), sum(up) / len(up)))

        # Fed-MENU: own-organ sub-encoder adapters + fusion + AGD + decoder
        g = groups(MENUModel(checkpoint_path=None, n_organs=3))
        up = [total(g, lambda k, o=ORGANS.index(org): k[0] in ("shared", "decoder") or (k[0] == "organ" and k[2] == o))
              for _, org, _ in CLIENTS]
        rows.append(("Fed-MENU", total(g), sum(up) / len(up)))

        # FedSHA: adapters in the client's window + gates + scale/shift + decoder
        model = FedSHA(checkpoint_path=None)
        g = groups(model)
        per_adapter = total(g, lambda k: k[0] == "pool") / model.M
        up = []
        for _, org, mod in CLIENTS:
            model.set_organ(ORGANS.index(org))
            n_used = int(model.in_window().sum())
            up.append(n_used * per_adapter + total(g, lambda k: k[0] in ("gate", "film", "decoder")))
        rows.append(("FedSHA", total(g), sum(up) / len(up)))

    print(f"\n{'Method':34s} {'trainable (M)':>14s} {'upload / client / round (M)':>28s}")
    for name, tr, up in rows:
        print(f"{name:34s} {tr / M:14.2f} {up / M:28.2f}")


if __name__ == "__main__":
    main()