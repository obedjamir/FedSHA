"""
FedSHA server. Per round: local training -> aggregation -> evaluation.

Setup (once): every client sends the name of its organ. The server sorts the names and organ i gets the 2 x 2
organ region i of the 4 x 4 adapter grid. Each client builds its sliding window locally from its organ region
(grid centre in the early blocks -> organ region in the last blocks).

Aggregation (written back into the clients; the server never holds one global model):
  adapters      : --expert_agg gate (default): adapter e = weighted average over the clients,
                  weight = the client's gate mass on e in the round (0 outside its window);
                  window: n_c over the clients that have e in their window; fedavg: n_c over all clients.
  gates         : row e of every gate averaged with the weights of adapter e.
  gamma / beta  : n_c over all clients.
  mask decoder  : n_c over all clients. Mask tokens and their hypernetwork MLPs stay local.
n_c = number of training volumes of client c.
"""
import os

import torch

from flcore.clients.clientFedSHA import clientFedSHA
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import param_group


class FedSHAServer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, {self.args.model_name}, num_clients:{args.num_clients}"
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp.replace(', ', '-')}-{args.goal}-{self.times}.h5")
        print(f"Adapter aggregation: {args.expert_agg}")

        self.set_clients(args, clientFedSHA)
        torch.cuda.empty_cache()
        self.place_organs()
        if getattr(args, "slices_per_epoch", 0) < 0:
            args.slices_per_epoch = min(c.train_size for c in self.clients)
        print("Train slices (volumes): " + ", ".join(f"client {c.id}: {c.train_size} ({c.n_volumes})" for c in self.clients)
              + f" | per epoch: {args.slices_per_epoch}")
        print("Finished creating server and clients.")

    def place_organs(self):
        """Sorted organ names -> organ regions of the grid."""
        organs = sorted({c.organ_key for c in self.clients})
        print("Organ regions: " + ", ".join(f"{k[:6]} -> {i}" for i, k in enumerate(organs)))
        for c in self.clients:
            c.grid_place = organs.index(c.organ_key)
            c.ensure_organ()

    def adapter_weights(self):
        """{client id: (M,) weight of every adapter} for the chosen --expert_agg."""
        mode = self.args.expert_agg
        w = {}
        for c in self.clients:
            n = float(c.n_volumes)
            win = c.model.in_window().float()
            if mode == "fedavg":
                w[c.id] = torch.full_like(win, n)
            elif mode == "window":
                w[c.id] = n * win
            else:                                                      # gate: measured gate mass, no n_c
                g = c.gate_mass
                w[c.id] = (g.float().cpu() if g is not None else torch.ones_like(win)) * win
        if mode == "gate":
            print("  gate mass per adapter: " + " | ".join(
                f"c{c.id}: " + " ".join(f"{float(v):.3f}" for v in w[c.id].tolist()) for c in self.clients), flush=True)
        return w

    @torch.no_grad()
    def aggregate(self):
        clients = self.clients
        n = {c.id: float(c.n_volumes) for c in clients}
        w = self.adapter_weights()
        skipped = set()
        for name, p in clients[0].model.named_parameters():
            if not p.requires_grad:
                continue
            g, _, idx = param_group(name)
            if g == "pool":
                if not self.average_into(name, [(c, w[c.id][idx]) for c in clients], clients):
                    skipped.add(idx)
            elif g == "gate":                                          # row e follows adapter e
                for e in range(clients[0].model.M):
                    self.average_into(name, [(c, w[c.id][e]) for c in clients], clients, row=e)
            elif g in ("film", "decoder"):
                self.average_into(name, [(c, n[c.id]) for c in clients], clients)
        if skipped:
            print(f"  adapters in no window (not aggregated): {sorted(skipped)}")

    def _ckpt(self, client, suffix="", goal=None):
        a = self.args
        goal = a.goal if goal is None else goal
        return f"{a.ckpt_dir}/{a.dataset}-{a.model_name}-num_clients:{a.num_clients}-{client.id}-{goal}-{self.times}{suffix}.pth"

    def finetune(self, src_goal):
        """FedSHA-P: every client starts from the FedSHA model of run `src_goal` (best checkpoint) and fine-tunes
        the adapters, gates, gamma / beta and mask decoder on its own data for --ft_epochs epochs, with the same
        window as in FL. The loaded model (epoch 0) and every epoch are validated, and each client's best epoch
        is tested."""
        a = self.args
        E = int(a.ft_epochs)
        print(f"\n==> FedSHA-P: fine-tuning run '{src_goal}' for {E} epochs", flush=True)
        for c in self.clients:
            c.model.load_state_dict(torch.load(self._ckpt(c, goal=src_goal), map_location="cpu"))
            c.model.to(a.device)
            for nm, p in c.model.named_parameters():
                p.requires_grad = p.requires_grad and param_group(nm)[0] in ("pool", "gate", "film", "decoder", "slot", "tokens")
            params = [p for p in c.model.parameters() if p.requires_grad]
            c.optimizer = torch.optim.AdamW([{"params": params, "lr": c.lrs["dec"], "base_lr": c.lrs["dec"],
                                              "name": "ft", "weight_decay": 0.0}])
            c.local_steps = 1
        n_tr = sum(p.numel() for p in self.clients[0].model.parameters() if p.requires_grad)
        print(f"  trainable per client: {n_tr / 1e6:.2f}M", flush=True)
        self.client_best, self.non_improve_rounds = {}, 0
        self.evaluate()                                                  # epoch 0 = the FedSHA model
        self.select_checkpoints(None, -1, lambda c: self._ckpt(c))
        for ep in range(E):
            print(f"\n------------- Fine-tuning epoch: [{ep + 1:3d}/{E}]-------------", flush=True)
            for c in self.clients:
                c.current_round = ep
                c.model.train()
                c.run_epochs(tag="ft ")
            self.evaluate()
            self.save_results(fn=self.hist_result_fn)
            self.select_checkpoints(None, ep, lambda c: self._ckpt(c))
        self.test_best()

    def test_best(self):
        print("==> Test with each client's best checkpoint...", flush=True)
        for c in self.clients:
            c.model.load_state_dict(torch.load(self._ckpt(c), map_location="cpu"))
        self.evaluate(val=False)
        self.save_results(fn=self.hist_result_fn)

    def train(self):
        if getattr(self.args, "ft_from", ""):
            self.finetune(self.args.ft_from)
            return
        if self.args.prev_round > 0:
            print("Loading Previous Checkpoint...")
            for c in self.clients:
                c.current_round = self.args.prev_round
                c.model.load_state_dict(torch.load(self._ckpt(c, "-ckpt"), map_location="cpu"))
                c.model.to(self.args.device)

        for i in range(self.args.prev_round, self.global_rounds):
            print(f"\n------------- Round number: [{i + 1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.clients)} clients...", flush=True)
            for c in self.clients:
                c.current_round = i
                c.train()
            print("==> Aggregation", flush=True)
            self.aggregate()

            print("==> Evaluating personalized models...", flush=True)
            mean_val, _ = self.evaluate()
            self.save_results(fn=self.hist_result_fn)
            self.select_checkpoints(mean_val, i, lambda c: self._ckpt(c))
            for c in self.clients:
                torch.save(c.model.state_dict(), self._ckpt(c, "-ckpt"))
            if self.should_stop(i):
                break
        self.test_best()
