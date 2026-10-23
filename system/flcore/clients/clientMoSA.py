"""
FedMoSA client (competing method), ported onto the common setup of this federation (frozen MedSAM, 2.5D inputs,
focal + Dice loss, AdamW schedule, slices per epoch, per-client model selection - identical for every method).
Kept from FedMoSA:
  * modality-wise batching: every batch holds one modality (slices grouped by modality, shuffled within a modality,
    incomplete batches dropped, batches of the modalities interleaved starting with the largest; as MergedDataset);
    the model is switched to the batch's modality before the forward pass (set_modality)
  * specialised adapter of the batch's modality + shared adapter in every block (MoSAModel)
  * prototype alignment: for each of the 12 blocks x 2 branches, the batch mean of the shared adapter's output
    (with gradient) is pulled to the global prototype:  L_proto = mean_{blocks, branches} || p_batch - p_global ||_2,
    total loss = L_seg + lam_proto * L_proto; off until a global prototype exists (first round)
  * local prototype = mean of the (detached) batch prototypes of the round's training batches; sent every round
"""
import torch
from torch.utils.data import DataLoader

from flcore.clients.clientseg import clientSeg
from flcore.clients.clientbase import SliceDataset


class ModalityBatchSampler:
    """Batches that each hold one modality (FedMoSA's MergedDataset order). modalities: one entry per item."""
    def __init__(self, modalities, batch_size, shuffle, drop_last, seed=0):
        self.mods, self.bs, self.shuffle, self.drop_last = list(modalities), batch_size, shuffle, drop_last
        self.gen = torch.Generator().manual_seed(seed)
        self.batch_modality = []

    def _batches(self):
        by_mod = {}
        for i, m in enumerate(self.mods):
            by_mod.setdefault(m, []).append(i)
        batches = {}
        for m, idx in by_mod.items():
            if self.shuffle:
                idx = [idx[j] for j in torch.randperm(len(idx), generator=self.gen).tolist()]
            n = len(idx) // self.bs if self.drop_last else -(-len(idx) // self.bs)
            batches[m] = [idx[k * self.bs:(k + 1) * self.bs] for k in range(n)]
        order = sorted(batches, key=lambda m: -len(batches[m]))       # modality with most batches first
        out, ptr = [], {m: 0 for m in order}
        while any(ptr[m] < len(batches[m]) for m in order):          # strict alternation
            for m in order:
                if ptr[m] < len(batches[m]):
                    out.append((m, batches[m][ptr[m]]))
                    ptr[m] += 1
        return out

    def __iter__(self):
        plan = self._batches()
        self.batch_modality = [m for m, _ in plan]
        return iter([b for _, b in plan])

    def __len__(self):
        return len(self._batches())


class clientMoSA(clientSeg):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.lam_proto = float(getattr(args, "lam_proto_mosa", 0.01))
        self.global_protos = None            # [(attn, mlp)] per block, from the server
        self.proto_sum, self.proto_count = None, 0
        self.n_modalities = 1

    # ---- modality of every slice (one modality per client in this federation: from the client's meta) ----
    def item_modality(self, item):
        return self.modality

    def _loader(self, items, batch_size, shuffle, drop_last):
        nw = int(getattr(self.args, "num_workers", 8))
        kw: dict = dict(num_workers=nw, pin_memory=str(self.device).startswith("cuda"))
        if nw > 0:
            kw["prefetch_factor"] = 4
        sampler = ModalityBatchSampler([self.item_modality(it) for it in items], batch_size, shuffle, drop_last,
                                       seed=hash((self.id, self.current_round, self.current_step)) % (2 ** 31))
        loader = DataLoader(SliceDataset(items, self.img_size, self.mask_size), batch_sampler=sampler, **kw)
        loader.modality_sampler = sampler
        return loader

    def on_batch(self, loader, b):
        self.model.set_modality(loader.modality_sampler.batch_modality[b])

    # ---- prototype alignment ----
    def proto_loss(self, out, y):
        protos = self.model.batch_prototypes()
        with torch.no_grad():                                  # local prototype: mean of the batch prototypes
            cur = [torch.stack([a.detach(), m.detach()]) for a, m in protos]
            cur = torch.stack(cur).cpu()                       # (L, 2, C)
            self.proto_sum = cur if self.proto_sum is None else self.proto_sum + cur
            self.proto_count += 1
        if self.global_protos is None or self.lam_proto <= 0:
            return None
        G = self.global_protos.to(protos[0][0].device)
        losses = [torch.norm(p - G[l, j], p=2) for l, pair in enumerate(protos) for j, p in enumerate(pair)]
        return self.lam_proto * torch.stack(losses).mean()

    def local_prototype(self):
        if self.proto_sum is None:
            return None
        return self.proto_sum / self.proto_count

    def train(self):
        self.model.train()
        f = self.apply_lr(self.optimizer)
        self.proto_sum, self.proto_count = None, 0
        print(f"Client {self.id} | {self.modality} {self.organ} | lr x{f:.2f} | FedMoSA | prototype "
              f"{'on' if self.global_protos is not None and self.lam_proto > 0 else 'off'}")
        self.model.set_collect(True)
        try:
            self.run_epochs(self.proto_loss)
        finally:
            self.model.set_collect(False)                      # evaluation does not touch the prototypes