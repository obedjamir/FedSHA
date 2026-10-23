"""
Fed-MENU server (competing method; Xu et al., IEEE TMI 2023). FedAvg (weights = train size, Eq. 1 of the paper):
  sub-encoder of organ m (organ m's adapters) : averaged over the clients that label organ m (the only ones that
                                                 tune it); every client receives it
  fusion layer, decoder trunk, AGD             : averaged over all clients
  mask slots (mask tokens + hypernetwork MLPs) : local, as for every baseline
Organ index of a client: position of its anonymous organ key among the sorted keys (as for our method).
"""
import os
from collections import defaultdict

import torch

from flcore.clients.clientMENU import clientMENU
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import param_group


class FedMENU(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, {self.args.model_name}, num_clients:{args.num_clients}"
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp.replace(', ', '-')}-{args.goal}-{self.times}.h5")
        self.set_clients(args, clientMENU)
        self.global_model = None
        torch.cuda.empty_cache()
        keys = sorted({c.organ_key for c in self.clients})
        assert len(keys) <= self.clients[0].model.n_organs, \
            f"{len(keys)} organs but the model has {self.clients[0].model.n_organs} sub-encoders (--menu_organs)"
        for c in self.clients:
            c.organ_index = keys.index(c.organ_key)
            c.ensure_organ()
        if getattr(args, "slices_per_epoch", 0) < 0:
            args.slices_per_epoch = min(c.train_size for c in self.clients)
        print("Fed-MENU | sub-encoders: " + ", ".join(f"client {c.id} ({c.organ}) -> {c.organ_index}" for c in self.clients)
              + f" | lambda_aux {args.lam_menu_aux}")
        print("Finished creating server and clients.")

    def _ckpt(self, client, suffix=""):
        a = self.args
        return f"{a.ckpt_dir}/{a.dataset}-{a.model_name}-num_clients:{a.num_clients}-{client.id}-{a.goal}-{self.times}{suffix}.pth"

    @torch.no_grad()
    def aggregate(self):
        clients = self.clients
        n = {c.id: float(c.n_volumes) for c in clients}                    # weight = training volumes
        for name, p in clients[0].model.named_parameters():
            if not p.requires_grad:
                continue
            g, _, idx = param_group(name)
            if g == "organ":                                    # sub-encoder of organ idx
                self.average_into(name, [(c, n[c.id] if c.organ_index == idx else 0.0) for c in clients], clients)
            elif g in ("shared", "decoder"):                    # fusion layer, AGD, decoder trunk
                self.average_into(name, [(c, n[c.id]) for c in clients], clients)
        groups = defaultdict(list)
        for c in clients:
            groups[c.organ_index].append(c.id)
        print("  sub-encoders averaged per organ: " + ", ".join(f"{m}: {v}" for m, v in sorted(groups.items()))
              + " | fusion + decoder + AGD over all clients", flush=True)

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