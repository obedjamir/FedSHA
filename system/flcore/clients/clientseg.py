"""
clientSeg: everything shared by every method - labels, losses, optimiser, learning-rate schedule, local training
loop and evaluation (per-case Dice).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.clients.clientbase import Client
from flcore.trainmodel.models import organ_key, modality_key
from utils.data_utils import read_client_meta


class DiceLoss(nn.Module):
    """Soft Dice per image and channel, averaged over the (image, channel) pairs that contain the class."""
    def forward(self, inputs, targets, smooth=1.):
        p = torch.sigmoid(inputs).flatten(2)
        t = targets.flatten(2)
        present = t.sum(-1) > 0
        if not present.any():
            return inputs.sum() * 0.0
        dice = (2. * (p * t).sum(-1) + smooth) / (p.sum(-1) + t.sum(-1) + smooth)
        return 1 - dice[present].mean()


class FocalBCE(nn.Module):
    """Sigmoid focal loss (Lin et al. 2017), mean over pixels."""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.gamma == 0:
            return bce.mean()
        return ((1 - torch.exp(-bce)) ** self.gamma * bce).mean()


class clientSeg(Client):
    LR_DEFAULT = {"enc": 1e-4, "dec": 1e-4}

    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.args = args
        self.current_step = 0
        self.dice = DiceLoss()
        self.seg_bce = FocalBCE(getattr(args, "focal_gamma", 2.0))
        self.bce_scaling, self.dice_scaling = args.lam_bce, args.lam_dice

        # ---- labels / organ / modality of this client (written by the dataset generator) ----
        self.meta = read_client_meta(self.dataset, self.id, self.dataset_id)
        self.seg_regions = [list(r) for r in self.meta.get("seg_regions", [[1]])]
        self.seg_classes = list(self.meta.get("seg_classes", ["fg"] * len(self.seg_regions)))
        self.organ = str(self.meta.get("organ", "organ"))
        self.modality = str(self.meta.get("modality", "CT"))
        self.organ_key = organ_key(self.organ)            # organ name, sent to the server
        self.n_seg = len(self.seg_regions)
        self.model.configure_heads(n_seg=self.n_seg)
        self.modality_key = modality_key(self.modality)   # modality name, sent to the server
        self.grid_place = None                            # organ region of the grid, set by the server (FedSHA)
        print(f"Client {self.id} | {self.modality} {self.organ} | seg {self.seg_classes}")

        self.lrs = dict(self.LR_DEFAULT)
        print(f"Client {self.id} | lr " + ", ".join(f"{g} {v:.1e}" for g, v in self.lrs.items()))
        self.optimizer = self.build_optimizer([(n, p) for n, p in self.model.named_parameters() if p.requires_grad])

    # ------------------------------------------------------------
    # Optimisation (identical for every method): AdamW, no weight decay,
    # lr x (1 warm-up round, then cosine to 0.1x). Groups: enc (adapters / gates), dec (MedSAM decoder)
    # ------------------------------------------------------------
    @staticmethod
    def lr_group(name):
        return "dec" if name.startswith(("sam.mask_decoder", "sam.prompt_encoder")) else "enc"

    def build_optimizer(self, named_params):
        groups = {}
        for n, p in named_params:
            groups.setdefault(self.lr_group(n), []).append(p)
        return torch.optim.AdamW([{"params": ps, "lr": self.lrs[g], "base_lr": self.lrs[g], "name": g,
                                   "weight_decay": 0.0} for g, ps in groups.items()])

    def lr_factor(self):
        r, R = self.current_round or 0, self.args.global_rounds
        W = getattr(self.args, "warmup_rounds", 1)
        if r < W:
            return (r + 1) / (W + 1)
        prog = min((r - W) / max(R - W - 1, 1), 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    def apply_lr(self, optimizer):
        f = self.lr_factor()
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * f
        return f

    # ------------------------------------------------------------
    # Data / targets / loss
    # ------------------------------------------------------------
    def ensure_organ(self):
        """Builds the windows (FedSHA, once the server has sent the organ region); no-op for the baselines."""
        if self.grid_place is not None and getattr(self.model, "placed", True) is False:
            print(f"Client {self.id} | {self.modality} {self.organ} -> organ region {self.grid_place}")
            self.model.set_organ(self.grid_place)

    def make_targets(self, y_lab):
        """(B,1,H,W) label map -> (B, n_seg, H, W) float, one channel per region (slot)."""
        return torch.cat([torch.isin(y_lab, torch.tensor(r, device=y_lab.device, dtype=y_lab.dtype))
                          for r in self.seg_regions], 1).float()

    def prepare_batch(self, x, y):
        return x.to(self.device, non_blocking=True), self.make_targets(y.to(self.device, non_blocking=True))

    @staticmethod
    def seg_logits(out, target):
        """Decoder masks come out at img_size / 4: upsample the logits to the label resolution."""
        logits = out["seg"]
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def on_batch(self, loader, b):
        """Hook before the forward pass of batch b (FedMoSA: sets the batch's modality)."""
        pass

    def seg_loss(self, out, y):
        logits = self.seg_logits(out, y)
        return self.bce_scaling * self.seg_bce(logits, y) + self.dice_scaling * self.dice(logits, y)

    def run_epochs(self, extra_loss=None, tag=""):
        """local_steps epochs of local training on fresh random slices; extra_loss(out, y) -> tensor or None."""
        round_tot, round_nb = 0.0, 0
        for epoch_idx in range(self.local_steps):
            loader = self.load_train_data(epoch=f"{tag}{epoch_idx}" if tag else epoch_idx)
            tot, ext, nb = 0.0, 0.0, 0
            self.current_step += 1
            for b, (x, y, _, _) in enumerate(loader):
                x, y = self.prepare_batch(x, y)
                self.batch_x = x                                   # input of the current batch (for method hooks)
                self.on_batch(loader, b)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.seg_loss(out, y)
                if extra_loss is not None:
                    e = extra_loss(out, y)
                    if e is not None:
                        loss = loss + e
                        ext += float(e.detach())
                loss.backward()
                self.model.drop_unused_expert_grads()
                self.optimizer.step()
                tot += loss.item()
                nb += 1
            extra = f" | extra {ext / max(nb, 1):.4f}" if ext else ""
            print(f"Client {self.id} | {tag}epoch {epoch_idx + 1}: loss {tot / max(nb, 1):.4f}{extra}", flush=True)
            round_tot, round_nb = round_tot + tot, round_nb + nb
        self.model.clear_routing()
        return round_tot / max(round_nb, 1)                    # mean training loss of the round

    # ------------------------------------------------------------
    # Evaluation (identical for every method)
    # ------------------------------------------------------------
    def get_eval_model(self, temp_model=None):
        return self.model

    def test_metrics(self, temp_model=None, val=True):
        self.ensure_organ()
        loader = self.load_val_data() if val else self.load_test_data()
        model = self.get_eval_model(temp_model)
        model.eval()
        K = self.n_seg
        seg_dice, seg_iou = [[] for _ in range(K)], [[] for _ in range(K)]
        seg_global = np.zeros((K, 2))
        case_sums = {}                                             # case -> (K, 3): overlap, pred + true, true
        test_loss, nb = 0.0, 0
        with torch.no_grad():
            for b, (x, y, _, cases) in enumerate(loader):
                x, y_true = self.prepare_batch(x, y)
                self.on_batch(loader, b)
                out = model(x)
                test_loss += self.seg_loss(out, y_true).item()
                y_pred = (torch.sigmoid(self.seg_logits(out, y_true)) > 0.5).float()
                inter = (y_pred * y_true).flatten(2).sum(2)
                p_sum, t_sum = y_pred.flatten(2).sum(2), y_true.flatten(2).sum(2)
                dice = 2 * inter / (p_sum + t_sum).clamp(min=1)
                iou = inter / (p_sum + t_sum - inter).clamp(min=1)
                for k in range(K):
                    present = t_sum[:, k] > 0                      # Dice on the slices that contain class k
                    seg_dice[k].extend(dice[present, k].tolist())
                    seg_iou[k].extend(iou[present, k].tolist())
                seg_global[:, 0] += inter.sum(0).cpu().numpy()
                seg_global[:, 1] += (p_sum + t_sum).sum(0).cpu().numpy()
                stats = torch.stack([inter, p_sum + t_sum, t_sum], -1).cpu().numpy()     # (B, K, 3)
                for i, case in enumerate(cases):
                    case_sums[case] = case_sums.get(case, 0) + stats[i]
                nb += 1
        model.clear_routing()
        d_k = [float(np.mean(d)) if d else None for d in seg_dice]
        i_k = [float(np.mean(v)) if v else None for v in seg_iou]
        g_k = 2 * seg_global[:, 0] / np.maximum(seg_global[:, 1], 1.0)
        # per-case Dice: overlap and pixel counts pooled over all slices of a case (patient), one Dice per case,
        # averaged over the cases that contain class k
        c_k, c_std = [], []
        for k in range(K):
            v = [2 * s[k, 0] / max(s[k, 1], 1.0) for s in case_sums.values() if s[k, 2] > 0]
            c_k.append(float(np.mean(v)) if v else None)
            c_std.append(float(np.std(v)) if v else None)
        # client score per case = mean Dice over the classes present in that case; mean +- std over cases
        per_case, per_case_iou = [], []
        self.case_scores = {}                                      # case -> per-case Dice (significance tests)
        for case, s in case_sums.items():
            ks = [k for k in range(K) if s[k, 2] > 0]
            if ks:
                per_case.append(float(np.mean([2 * s[k, 0] / max(s[k, 1], 1.0) for k in ks])))
                self.case_scores[str(case)] = per_case[-1]
                per_case_iou.append(float(np.mean([s[k, 0] / max(s[k, 1] - s[k, 0], 1.0) for k in ks])))
        metrics = {}
        if K > 1:
            for name, d, g, cd in zip(self.seg_classes, d_k, g_k, c_k):
                metrics[f"dice_{name}"] = d if d is not None else 0.0
                metrics[f"dice_global_{name}"] = float(g)
                metrics[f"dice_case_{name}"] = cd if cd is not None else 0.0
        valid = [d for d in d_k if d is not None]
        metrics["dice"] = float(np.mean(valid)) if valid else 0.0
        valid = [v for v in i_k if v is not None]
        metrics["iou"] = float(np.mean(valid)) if valid else 0.0
        metrics["dice_global"] = float(np.mean(g_k))
        valid = [v for v in c_k if v is not None]
        metrics["dice_case"] = float(np.mean(valid)) if valid else 0.0
        metrics["dice_case_client_mean"] = float(np.mean(per_case)) if per_case else 0.0
        metrics["dice_case_client_std"] = float(np.std(per_case)) if per_case else 0.0
        metrics["iou_case_client_mean"] = float(np.mean(per_case_iou)) if per_case_iou else 0.0
        metrics["iou_case_client_std"] = float(np.std(per_case_iou)) if per_case_iou else 0.0
        metrics["n_cases"] = float(len(case_sums))
        self.task_metrics = metrics
        print(f"Client {self.id} ({self.modality} {self.organ}) -> " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        if val:
            self.last_val_loss = test_loss / max(nb, 1)          # validation loss of the current model
        # score used for model selection and the mean over clients: per-case Dice (mean over the client's cases)
        return metrics["dice_case_client_mean"], metrics["iou_case_client_mean"], test_loss / max(nb, 1), nb * self.batch_size


