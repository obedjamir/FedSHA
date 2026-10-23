"""
FedHCA2 (Lu et al., "FedHCA2: Towards Hetero-Client Federated Multi-Task Learning", CVPR 2024)
as a competing method in the modality x organ federation (segmentation only, one decoder per client).
Aggregation ported from the official code (github.com/innovator-zero/FedHCA2: aggregate.py, models/hyperweight.py):
    encoder  (dense MSA adapters)            : hyper conflict-averse aggregation, learnable per-client alpha
    decoder  (MedSAM decoder trunk, 1/client): hyper cross-attention aggregation, learnable layer-wise beta
    mask slots (mask tokens + hypernetwork MLPs): local, never aggregated (as for every baseline)
Local training: plain task loss on the common base network (clientFedAvg).
Hyperweights are updated by SGD with the official rule (grad = d new / d w . (last - current)).
Every client receives its own (personalised) encoder adapters and decoders.
Round loop as in the official train.py: local training -> hyperweight update -> aggregation ->
evaluation. There is NO head re-fitting step (heads keep their locally trained values).
"""
import os
import numpy as np
import torch
from scipy.optimize import minimize

from flcore.clients.clientHCA2 import clientHCA2
from flcore.trainmodel.models import param_group
from flcore.servers.serverbase import Server


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def flat(state, names):
    return torch.cat([state[n].reshape(-1).float() for n in names])


def unflat(vec, names, shapes):
    out, s = {}, 0
    for n, sh in zip(names, shapes):
        k = int(np.prod(sh))
        out[n] = vec[s:s + k].reshape(sh)
        s += k
    return out


def snapshot(model, names):
    p = dict(model.named_parameters())
    return {n: p[n].detach().cpu().clone() for n in names}


def load_into(model, values):
    p = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in values.items():
            p[n].data.copy_(v.to(p[n].device, p[n].dtype))


# ------------------------------------------------------------------
# FedHCA2  (ported from the official aggregate.py / hyperweight.py)
# ------------------------------------------------------------------
def ca_delta(deltas, ca_c, rescale=1):
    """Conflict-averse aggregated delta (get_ca_delta in the official code)."""
    N = len(deltas)
    grads = torch.stack(deltas).t()                     # [d, N]
    GG = grads.t().mm(grads).double().cpu()
    g0_norm = (GG.mean() + 1e-8).sqrt()
    x_start = np.ones(N) / N
    A, b = GG.numpy(), x_start.copy()
    c = (ca_c * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
                + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=tuple((0, 1) for _ in x_start),
                   constraints={"type": "eq", "fun": lambda x: 1 - sum(x)})
    ww = torch.tensor(res.x, dtype=grads.dtype)
    gw = (grads * ww.reshape(1, -1)).sum(1)
    lmbda = c / (gw.norm() + 1e-8)
    g = grads.mean(1) + lmbda * gw
    return g if rescale == 0 else (g / (1 + ca_c ** 2) if rescale == 1 else g / (1 + ca_c))


def cross_attention(q, kv):
    attn = torch.softmax((q @ kv.t()) * q.size(-1) ** -0.5, dim=-1)
    return attn @ kv


class FedHCA2State:
    """Learnable hyper aggregation weights (alpha per client, beta per decoder layer and decoder)."""
    def __init__(self, n_clients, alpha_init=0.1, beta_init=0.1, lr=0.01, ca_c=0.4):
        self.alpha = torch.full((n_clients,), alpha_init)
        self.beta = {}                    # layer name -> tensor(K decoders)
        self.beta_init, self.lr, self.ca_c = beta_init, lr, ca_c
        self.last_enc_dir = None          # d(new_enc_k)/d(alpha_k) of the previous aggregation
        self.last_dec_dir = None          # {(dec_idx, layer): d(new)/d(beta)} of the previous aggregation

    # ---- hyperweight update (update_hyperweight in the official code) ----
    def update(self, enc_start, enc_now, dec_start=None, dec_now=None):
        """SGD on alpha / beta with grad = d(output)/d(w) . (last - current), as in the official code."""
        if self.last_enc_dir is not None:
            for k in range(len(enc_start)):
                if 0.0 <= self.alpha[k] <= 1.0:          # clamp(0, 1) passes gradient only inside
                    grad = (self.last_enc_dir * (enc_start[k] - enc_now[k])).sum()
                    self.alpha[k] -= self.lr * grad
        if self.last_dec_dir is not None and dec_start is not None:
            for (i, layer), direction in self.last_dec_dir.items():
                b = self.beta[layer]
                if 0.0 <= b[i] <= 1.0:
                    grad = sum((direction[n] * (dec_start[i][n] - dec_now[i][n])).sum() for n in direction)
                    b[i] -= self.lr * grad

    # ---- encoder: hyper conflict-averse aggregation ----
    def aggregate_encoders(self, enc_start, enc_now):
        deltas = [now - start for start, now in zip(enc_start, enc_now)]
        g = ca_delta(deltas, self.ca_c)
        self.last_enc_dir = g
        a = self.alpha.clamp(0, 1)
        return [start + d + a[k] * g for k, (start, d) in enumerate(zip(enc_start, deltas))]

    # ---- decoders: hyper cross-attention aggregation (layer-wise) ----
    def aggregate_decoders(self, dec_start, dec_now, layer_of):
        """dec_start / dec_now: list over decoders of {local param name -> tensor} (same names for all).
        layer_of: param name -> layer name (weight and bias of a layer share a beta)."""
        K = len(dec_start)
        names = list(dec_start[0].keys())
        delta = [{n: dec_now[i][n] - dec_start[i][n] for n in names} for i in range(K)]
        new = [{n: dec_start[i][n].clone() for n in names} for i in range(K)]
        self.last_dec_dir = {}
        for n in names:
            layer = layer_of[n]
            if layer not in self.beta:
                self.beta[layer] = torch.full((K,), self.beta_init)
            b = self.beta[layer].clamp(0, 1)
            kv = torch.stack([delta[j][n].reshape(-1) for j in range(K)])
            for i in range(K):
                ca = cross_attention(delta[i][n].reshape(1, -1), kv).reshape(delta[i][n].shape)
                new[i][n] += delta[i][n] + b[i] * ca
                self.last_dec_dir.setdefault((i, layer), {})[n] = ca
        return new


# ------------------------------------------------------------------
# Server
# ------------------------------------------------------------------
class FedHCA2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, {self.args.model_name}, num_clients:{args.num_clients}"
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp.replace(', ', '-')}-{args.goal}-{self.times}.h5")
        self.hca2 = None
        print(f"FedHCA2 | ca_c {args.hca2_ca_c} | alpha0 {args.hca2_alpha_init} | beta0 {args.hca2_beta_init} | hyper lr {args.hca2_lr}")
        self.set_clients(args, clientHCA2)
        self.global_model = None
        torch.cuda.empty_cache()
        if getattr(args, "slices_per_epoch", 0) < 0:
            args.slices_per_epoch = min(c.train_size for c in self.clients)
        m = self.clients[0].model
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        self.enc_names = [n for n in trainable if param_group(n)[0] == "shared"]       # MSA adapters
        self.dec_names = [n for n in trainable if param_group(n)[0] == "decoder"]      # decoder trunk
        print("Train slices: " + ", ".join(f"client {c.id}: {c.train_size}" for c in self.clients)
              + f" | per epoch: {args.slices_per_epoch}")
        print("Finished creating server and clients.")

    def _ckpt(self, client, suffix=""):
        a = self.args
        return f"{a.ckpt_dir}/{a.dataset}-{a.model_name}-num_clients:{a.num_clients}-{client.id}-{a.goal}-{self.times}{suffix}.pth"

    def snapshot_all(self):
        return {"enc": [flat(snapshot(c.model, self.enc_names), self.enc_names) for c in self.clients],
                "dec": [snapshot(c.model, self.dec_names) for c in self.clients]}

    def aggregate(self, start, now):
        p0 = dict(self.clients[0].model.named_parameters())
        shapes = [p0[n].shape for n in self.enc_names]
        if self.hca2 is None:
            self.hca2 = FedHCA2State(len(self.clients), self.args.hca2_alpha_init,
                                     self.args.hca2_beta_init, self.args.hca2_lr, self.args.hca2_ca_c)
        self.hca2.update(start["enc"], now["enc"], start["dec"], now["dec"])   # hyperweights (from round 2)
        new_enc = self.hca2.aggregate_encoders(start["enc"], now["enc"])
        layer_of = {n: n.rsplit(".", 1)[0] for n in self.dec_names}
        new_dec = self.hca2.aggregate_decoders(start["dec"], now["dec"], layer_of)
        for k, c in enumerate(self.clients):                                 # personalised encoder + decoder
            load_into(c.model, unflat(new_enc[k], self.enc_names, shapes))
            load_into(c.model, new_dec[k])
        print("FedHCA2 alpha: " + ", ".join(f"{a:.3f}" for a in self.hca2.alpha.tolist()), flush=True)

    def train(self):
        if self.args.prev_round > 0:
            print("Loading Previous Checkpoint... (hyperweights restart from their initial values)")
            for c in self.clients:
                c.current_round = self.args.prev_round
                c.model.load_state_dict(torch.load(self._ckpt(c, "-ckpt"), map_location="cpu"))
                c.model.to(self.args.device)
        for i in range(self.global_rounds):
            if i < self.args.prev_round:
                continue
            print(f"\n------------- Round number: [{i + 1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.clients)} clients...", flush=True)
            start = self.snapshot_all()
            for c in self.clients:
                c.current_round = i
                c.train()
            print("==> FedHCA2 aggregation", flush=True)
            self.aggregate(start, self.snapshot_all())
            print("==> Evaluating personalized models...", flush=True)
            mean_val, _ = self.evaluate()
            self.save_results(fn=self.hist_result_fn)
            self.select_checkpoints(mean_val, i, lambda c: self._ckpt(c))
            for c in self.clients:
                torch.save(c.model.state_dict(), self._ckpt(c, "-ckpt"))
            if self.should_stop(i):
                break
        print("==> Test with each client's best checkpoint...", flush=True)
        for c in self.clients:
            c.model.load_state_dict(torch.load(self._ckpt(c), map_location="cpu"))
        self.evaluate(val=False)
        self.save_results(fn=self.hist_result_fn)
