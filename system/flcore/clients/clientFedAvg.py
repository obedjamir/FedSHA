"""
Clients of the baselines on the common base network (frozen MedSAM + dense MSA adapters + MedSAM decoder).
Losses, optimiser, learning rates and evaluation come from clientSeg, so every method is scored the same way.
    local  : local training only (no communication)
    fedavg : plain local training of the received body                               (McMahan et al., 2017)
    fedprox: + mu/2 ||w - w_global||^2 over the body                                 (Li et al., 2020)
    fedrep : FedAvg of the body; the head (output layer: mask tokens + hypernetwork MLPs) stays local (Collins et al., 2021)
             local training per round: first the head with the body frozen (--rep_head_epochs, default 1), then the
             body with the head frozen (the remaining epochs), same total epochs as every other method
    moon   : (Li et al., 2021) + mu * model-contrastive loss on the encoder representation z (neck output, global
             average pooled): -log( e^{sim(z,z_g)/tau} / (e^{sim(z,z_g)/tau} + e^{sim(z,z_p)/tau}) ), with z_g from
             the received global model and z_p from the client's previous local model (no grad); mu = 1, tau = 0.5.
             Body aggregated with FedAvg.
    ditto  : two separate models per client, each with its own optimiser:            (Li et al., 2021)
             (1) global copy w (received global body + its own local mask slots): trained on the task loss;
                 its body is uploaded
             (2) personal model v (body + mask slots): trained on the task loss + lam/2 ||v_body - w^t||^2, with
                 w^t the received global body; v is the model that is evaluated and checkpointed
    fedala : (Zhang et al., AAAI 2023) Adaptive Local Aggregation. Before local training, the client mixes the
             received global body into its local body, theta <- theta + (theta_g - theta) * W, with element-wise
             weights W in [0, 1] learned on a random part of its training data (lr --ala_eta, --ala_rand_percent of
             one epoch). W is learned for the "higher" parameters (--ala_layers: decoder trunk by default); the other
             body parameters are overwritten by the global model. The first ALA (round 2) runs until the loss
             settles (std of the last --ala_num_pre_loss losses < --ala_threshold, at most --ala_max_epochs epochs);
             later rounds use one pass and keep W. Body aggregated with FedAvg; head local.
    feddbe : (Zhang et al., NeurIPS 2023) Domain Bias Eliminator. (1) Personalized representation bias memory
             (PRBM): a local learnable vector added to the encoder representation before the decoder, never sent.
             (2) Mean regularization (MR): kappa * MSE(zbar, zbar_g), with zbar the batch mean of the
             (pooled) representation before the PRBM and zbar_g the global mean representation (FedAvg of the
             clients' running means, momentum --dbe_momentum). Body aggregated with FedAvg; head and PRBM local.
Body = MSA adapters + decoder trunk (FedAvg over all clients). The mask slots (mask tokens + hypernetwork MLPs)
stay local: their meaning differs between clients (slot 0 = liver, kidney, nodule or tumor).
"""
import torch
import torch.nn.functional as F
from typing import Any

from flcore.clients.clientseg import clientSeg
from flcore.trainmodel.models import param_group

BODY_GROUPS = ("shared", "decoder")


class clientFedAvg(clientSeg):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.method = args.arch
        self.global_w: Any = None        # ditto: last received global body w^t
        self.g_state = None              # ditto: global copy w (all trainable parameters)
        self.p_state = None              # ditto: personal model v (all trainable parameters)
        self.upload = None               # ditto: trained body of the global copy, read by the server
        self.opt_personal = None         # ditto: separate optimiser of the personal model
        self.prev_enc = None             # moon: encoder-side body of the previous local model
        self.ala_W = None                # fedala: learned aggregation weights (name -> tensor in [0, 1])
        self.ala_start = True            # fedala: first ALA runs until the loss settles
        self.dbe_mean = None             # feddbe: running mean representation (uploaded)
        self.dbe_global = None           # feddbe: global mean representation (set by the server)

    def body_names(self):
        return [n for n, p in self.model.named_parameters() if p.requires_grad and param_group(n)[0] in BODY_GROUPS]

    def get_body(self):
        p = dict(self.model.named_parameters())
        return {n: p[n].detach().clone() for n in self.body_names()}

    def set_body(self, d):
        p = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, v in d.items():
                p[n].copy_(v.to(p[n].device, p[n].dtype))

    def trainable_names(self):
        return [n for n, p in self.model.named_parameters() if p.requires_grad]

    def get_state(self, names):
        p = dict(self.model.named_parameters())
        return {n: p[n].detach().clone() for n in names}

    # ---- MOON ----
    def enc_names(self):
        return [n for n in self.body_names() if param_group(n)[0] == "shared"]

    @torch.no_grad()
    def _rep_with(self, state, x):
        """Encoder representation of x with the encoder-side body `state` swapped in (restored afterwards)."""
        params = dict(self.model.named_parameters())
        keep = {n: params[n].detach().clone() for n in state}
        for n, v in state.items():
            params[n].copy_(v.to(params[n].device))
        z = self.model.encode(x).mean(dim=(2, 3))
        for n, v in keep.items():
            params[n].copy_(v)
        return z

    def on_batch(self, loader, b):
        """MOON: representations of the batch under the global and the previous model, computed before the
        forward pass of the local model (the parameters are swapped in place and restored)."""
        super().on_batch(loader, b)
        if self.method == "moon" and getattr(self, "_moon_glob", None) is not None:
            prev = self.prev_enc if self.prev_enc is not None else self._moon_glob
            self._zg = self._rep_with(self._moon_glob, self.batch_x)
            self._zp = self._rep_with(prev, self.batch_x)

    def _moon_loss(self, glob):
        mu, tau = float(getattr(self.args, "moon_mu", 1.0)), float(getattr(self.args, "moon_tau", 0.5))
        self._moon_glob = glob

        def loss(out, y):
            z = out["emb"].mean(dim=(2, 3))
            zg, zp = self._zg, self._zp
            pos = torch.nn.functional.cosine_similarity(z, zg, dim=1) / tau
            neg = torch.nn.functional.cosine_similarity(z, zp, dim=1) / tau
            return mu * (torch.logsumexp(torch.stack([pos, neg], 1), 1) - pos).mean()
        return loss

    def _prox(self, anchor, weight):
        """mu/2 ||w - anchor||^2 over the body (None if weight is 0)."""
        if anchor is None or weight <= 0:
            return None
        params = dict(self.model.named_parameters())
        anchor = {n: v.to(params[n].device) for n, v in anchor.items()}
        return lambda out, y: 0.5 * weight * sum(((params[n] - anchor[n]) ** 2).sum() for n in anchor)

    # ---- FedALA ----
    def ala_init(self, glob):
        """Adaptive local aggregation of the received global body `glob` into the local body (in place)."""
        a = self.args
        params = dict(self.model.named_parameters())
        names = list(glob)
        if all(torch.equal(params[n].detach().cpu(), glob[n].to(params[n].dtype)) for n in names[:1]):
            return                                                   # nothing to adapt (identical start)
        if self.current_round is None or self.current_round < 1:     # round 1: plain overwrite (as in the paper)
            self.set_body(glob)
            return
        high = [n for n in names if a.ala_layers == "body" or param_group(n)[0] == "decoder"]
        low = [n for n in names if n not in high]
        self.set_body({n: glob[n] for n in low})                     # lower parameters: global model
        dev = params[high[0]].device
        loc = {n: params[n].detach().clone() for n in high}
        diff = {n: glob[n].to(dev, params[n].dtype) - loc[n] for n in high}
        if self.ala_W is None:
            self.ala_W = {n: torch.ones_like(loc[n]) for n in high}
        W = self.ala_W
        eta = float(a.ala_eta)

        def apply():
            with torch.no_grad():
                for n in high:
                    params[n].copy_(loc[n] + diff[n] * W[n])

        saved = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for n, p in self.model.named_parameters():
            p.requires_grad = n in high
        self.model.train()
        apply()
        losses, ep = [], 0
        grad_ctx = torch.enable_grad()                               # aggregate() runs under torch.no_grad()
        grad_ctx.__enter__()
        try:
            while True:
                loader = self.load_train_data(epoch=f"ala{ep}")
                n_b = max(1, int(len(loader) * float(a.ala_rand_percent)))
                for b, (x, y, _, _) in enumerate(loader):
                    if b >= n_b:
                        break
                    x, y = self.prepare_batch(x, y)
                    self.batch_x = x
                    self.on_batch(loader, b)
                    self.model.zero_grad(set_to_none=True)
                    loss = self.seg_loss(self.model(x), y)
                    loss.backward()
                    with torch.no_grad():
                        for n in high:
                            g = params[n].grad
                            if g is not None:
                                W[n].sub_(eta * g * diff[n]).clamp_(0, 1)
                    apply()
                    losses.append(loss.item())
                ep += 1
                if not self.ala_start:
                    break
                k = int(a.ala_num_pre_loss)
                if (len(losses) > k and float(torch.tensor(losses[-k:]).std()) < a.ala_threshold) \
                        or ep >= int(a.ala_max_epochs):
                    break
        finally:
            grad_ctx.__exit__(None, None, None)
            self.model.zero_grad(set_to_none=True)
            for n, p in self.model.named_parameters():
                p.requires_grad = saved[n]
        print(f"Client {self.id} | ALA: {ep} epoch(s), {len(losses)} steps, loss {losses[-1]:.4f}, "
              f"mean W {float(torch.cat([w.flatten() for w in W.values()]).mean()):.3f}", flush=True)
        self.ala_start = False

    # ---- FedDBE ----
    def _dbe_loss(self):
        kappa, mom = float(self.args.dbe_kappa), float(self.args.dbe_momentum)

        def loss(out, y):
            z = out["emb"].mean(dim=(2, 3)).mean(0)                  # batch mean of the pooled representation
            with torch.no_grad():
                zd = z.detach()
                self.dbe_mean = zd.clone() if self.dbe_mean is None else (1 - mom) * self.dbe_mean + mom * zd
            if self.dbe_global is None:
                return None
            return kappa * F.mse_loss(z, self.dbe_global.to(z.device))       # kappa * MSE, as in the paper
        return loss

    def train(self):
        self.model.train()
        f = self.apply_lr(self.optimizer)
        print(f"Client {self.id} | {self.modality} {self.organ} | lr x{f:.2f} | {self.method}")
        if self.method == "fedrep":
            head = {n for n, p in self.model.named_parameters()                 # output layer (local)
                    if p.requires_grad and param_group(n)[0] in ("slot", "tokens")}
            body = set(self.body_names())
            saved = {n: p.requires_grad for n, p in self.model.named_parameters()}
            steps = self.local_steps
            n_head = min(max(1, int(getattr(self.args, "rep_head_epochs", 1))), steps - 1) if steps > 1 else 1
            try:
                for phase, n_ep, train_set in (("head ", n_head, head), ("body ", max(steps - n_head, 1), body)):
                    for n, p in self.model.named_parameters():
                        p.requires_grad = saved[n] and n in train_set
                    self.local_steps = n_ep
                    self.run_epochs(tag=phase)
            finally:
                self.local_steps = steps
                for n, p in self.model.named_parameters():
                    p.requires_grad = saved[n]
        elif self.method == "moon":
            glob = {n: v.detach().clone() for n, v in self.get_state(self.enc_names()).items()}   # received model
            try:
                self.run_epochs(self._moon_loss(glob))
            finally:
                self._moon_glob = None
            self.prev_enc = self.get_state(self.enc_names())
        elif self.method == "fedprox":
            self.run_epochs(self._prox(self.get_body(), float(self.args.prox_mu)))
        elif self.method == "ditto":
            names = self.trainable_names()
            if self.g_state is None:                               # round 1: w and v start from the same model
                self.g_state = self.get_state(names)
                self.p_state = {n: v.clone() for n, v in self.g_state.items()}
                self.opt_personal = self.build_optimizer(
                    [(n, p) for n, p in self.model.named_parameters() if p.requires_grad])
            if self.global_w is not None:                          # received global body -> global copy
                self.g_state.update({n: v.clone() for n, v in self.global_w.items()})
            anchor = {n: self.g_state[n].clone() for n in self.body_names()}     # w^t
            self.set_body(self.g_state)                            # (1) global copy, own optimiser
            self.run_epochs(tag="global ")
            self.g_state = self.get_state(names)
            self.upload = {n: self.g_state[n] for n in self.body_names()}
            self.set_body(self.p_state)                            # (2) personal model, own optimiser
            opt_g = self.optimizer
            self.optimizer = self.opt_personal
            self.apply_lr(self.optimizer)
            try:
                self.run_epochs(self._prox(anchor, float(self.args.ditto_lambda)), tag="personal ")
            finally:
                self.optimizer = opt_g
            self.p_state = self.get_state(names)                   # stays loaded: evaluated and checkpointed
        elif self.method == "feddbe":
            self.dbe_mean = None
            self.run_epochs(self._dbe_loss())
        else:                                                      # local / fedavg / fedala (ALA in ala_init)
            self.run_epochs()