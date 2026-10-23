"""
Baselines on the common base network (frozen MedSAM + dense MSA adapters + MedSAM decoder).
The body (MSA adapters + decoder trunk) is FedAvg'd over ALL clients, weighted by train-set size;
the mask slots (mask tokens + hypernetwork MLPs) stay local.
    FedAvg  (McMahan et al., 2017)
    FedProx (Li et al., 2020): + mu/2 ||w - w_global||^2 on the body during local training
    FedRep  (Collins et al., 2021): FedAvg of the body; head (mask slots) local; head-then-body local training
    MOON    (Li et al., 2021): FedAvg of the body; model-contrastive loss during local training
    FedALA  (Zhang et al., 2023): FedAvg of the body; each client mixes the global body into its own (ALA)
    FedDBE  (Zhang et al., 2023): FedAvg of the body; local representation bias + mean regularization
    Ditto   (Li et al., 2021): FedAvg of the clients' trained global copies; each client also keeps a personal
                               model regularised towards the global one, evaluated on the personal model
"""
import os
import torch

from flcore.clients.clientFedAvg import clientFedAvg
from flcore.servers.serverbase import Server


class FedAvgSeg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, {self.args.model_name}, num_clients:{args.num_clients}"
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp.replace(', ', '-')}-{args.goal}-{self.times}.h5")
        info = {"fedavg": "", "fedprox": f"mu {args.prox_mu}", "ditto": f"lambda {args.ditto_lambda}",
                "fedrep": f"head epochs {args.rep_head_epochs}",
                "moon": f"mu {args.moon_mu}, tau {args.moon_tau}",
                "fedala": f"ALA on {args.ala_layers}, eta {args.ala_eta}, rand {args.ala_rand_percent}",
                "feddbe": f"kappa {args.dbe_kappa}, momentum {args.dbe_momentum}"}[args.arch]
        print(f"{args.arch} | body (MSA adapters + decoder trunk) averaged over all clients | {info}")
        self.set_clients(args, clientFedAvg)
        self.global_model = None
        torch.cuda.empty_cache()
        if getattr(args, "slices_per_epoch", 0) < 0:
            args.slices_per_epoch = min(c.train_size for c in self.clients)
        print("Train slices (volumes): " + ", ".join(f"client {c.id}: {c.train_size} ({c.n_volumes})" for c in self.clients)
              + f" | per epoch: {args.slices_per_epoch}")
        print("Finished creating server and clients.")

    def _ckpt(self, client, suffix=""):
        a = self.args
        return f"{a.ckpt_dir}/{a.dataset}-{a.model_name}-num_clients:{a.num_clients}-{client.id}-{a.goal}-{self.times}{suffix}.pth"

    @torch.no_grad()
    def aggregate(self):
        names = self.clients[0].body_names()
        n_k = torch.tensor([float(c.n_volumes) for c in self.clients])     # weight = training volumes
        w = n_k / n_k.sum()
        if self.args.arch == "ditto":                                  # uploaded global copies
            W = [c.upload for c in self.clients]
        else:
            W = [dict(c.model.named_parameters()) for c in self.clients]
        avg = {n: torch.stack([wk * Wk[n].detach().float().cpu() for wk, Wk in zip(w, W)]).sum(0) for n in names}
        for c in self.clients:
            if self.args.arch == "ditto":
                c.global_w = {n: v.clone() for n, v in avg.items()}     # next round's global copy; v stays loaded
            elif self.args.arch == "fedala":
                c.ala_init({n: v.clone() for n, v in avg.items()})      # adaptive local aggregation
            else:
                c.set_body(avg)
        if self.args.arch == "feddbe":                                 # global mean representation
            ms = [(wk, c.dbe_mean) for wk, c in zip(w, self.clients) if c.dbe_mean is not None]
            if ms:
                g = sum(wk * m.float().cpu() for wk, m in ms) / sum(wk for wk, _ in ms)
                for c in self.clients:
                    c.dbe_global = g.clone()
        print(f"  body averaged over clients {[c.id for c in self.clients]}", flush=True)

    def train(self):
        if self.args.prev_round > 0:
            print("Loading Previous Checkpoint...")
            for c in self.clients:
                c.current_round = self.args.prev_round
                c.model.load_state_dict(torch.load(self._ckpt(c, "-ckpt"), map_location="cpu"))
                c.model.to(self.args.device)

        for i in range(self.global_rounds):
            if i < self.args.prev_round:
                continue
            print(f"\n------------- Round number: [{i + 1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.clients)} clients...", flush=True)
            for c in self.clients:
                c.current_round = i
                c.train()
            print("==> Aggregation", flush=True)
            self.aggregate()
            print("==> Evaluating models...", flush=True)
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