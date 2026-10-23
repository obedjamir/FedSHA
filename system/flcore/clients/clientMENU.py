"""
Fed-MENU client (competing method; Xu et al., "Federated Multi-organ Segmentation with Inconsistent Labels",
IEEE TMI 2023), ported to the frozen MedSAM (MENUModel): one sub-encoder per organ, concatenated and fused before
the decoder, plus an auxiliary generic decoder (AGD).
Local training (Eqs. 3 and 5 of the paper): the client tunes the sub-encoder of its own organ, the fusion layer and
decoder, the AGD and its mask slots; the other organs' sub-encoders run without gradient.
    L = L_sup + L_aux
    L_sup : segmentation loss on the client's own classes (focal + Dice, as for every method; the mask slots are
            local, so the partial-label marginal / exclusion losses of the paper's global multi-organ head reduce to it)
    L_aux : cross-entropy + Dice of the AGD's organ-agnostic binary mask (the client's labelled foreground) computed
            on the client's own sub-encoder features, upsampled to the label resolution
"""
import torch
import torch.nn.functional as F

from flcore.clients.clientseg import clientSeg


class clientMENU(clientSeg):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.lam_aux = float(getattr(args, "lam_menu_aux", 1.0))
        self.organ_index = None                              # set by the server (sorted anonymous organ keys)

    def ensure_organ(self):
        if self.organ_index is not None and self.model.own != self.organ_index:
            self.model.set_organ(self.organ_index)

    def aux_loss(self, out, y):
        if "agd" not in out or self.lam_aux <= 0:
            return None
        logits = F.interpolate(out["agd"], size=y.shape[-2:], mode="bilinear", align_corners=False)
        fg = y.amax(1).long()                                # organ-agnostic mask: any of the client's classes
        ce = F.cross_entropy(logits, fg)
        p = torch.softmax(logits, 1)[:, 1]
        t = fg.float()
        dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
        return self.lam_aux * (ce + dice.mean())

    def train(self):
        self.ensure_organ()
        self.model.train()
        f = self.apply_lr(self.optimizer)
        print(f"Client {self.id} | {self.modality} {self.organ} | lr x{f:.2f} | Fed-MENU | own sub-encoder {self.organ_index}")
        self.run_epochs(self.aux_loss)
