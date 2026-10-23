"""
FedMoSA server (competing method). Aggregation as in FedMoSA:
  modality-restricted : the specialised adapters of modality m are averaged over the clients holding m,
                        weighted by their number of slices of m; every client receives them
  modality-balanced   : shared adapters, decoder trunk and the shared prototypes are averaged over all clients,
                        weighted by (number of modalities of the client) x (train size)
Mask slots (mask tokens + hypernetwork MLPs) stay local, as for every baseline (their meaning differs per client).
Round loop and model selection are the same as for every other method.
"""
import os
import torch

from flcore.clients.clientMoSA import clientMoSA
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import param_group


class FedMoSA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, {self.args.model_name}, num_clients:{args.num_clients}"
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp.replace(', ', '-')}-{args.goal}-{self.times}.h5")
        self.set_clients(args, clientMoSA)
        self.global_model = None
        torch.cuda.empty_cache()
        if getattr(args, "slices_per_epoch", 0) < 0:
            args.slices_per_epoch = min(c.train_size for c in self.clients)
        print("FedMoSA | modalities: " + ", ".join(f"client {c.id}: {c.modality}" for c in self.clients)
              + f" | lambda_proto {args.lam_proto_mosa}")
        print("Finished creating server and clients.")

    def _ckpt(self, client, suffix=""):
        a = self.args
        return f"{a.ckpt_dir}/{a.dataset}-{a.model_name}-num_clients:{a.num_clients}-{client.id}-{a.goal}-{self.times}{suffix}.pth"

    @torch.no_grad()
    def aggregate(self):
        clients = self.clients
        n = {c.id: float(c.n_volumes) for c in clients}                # weight = training volumes (one modality per client)
        bal = {c.id: c.n_modalities * n[c.id] for c in clients}        # modality-balanced weight
        for name, p in clients[0].model.named_parameters():
            if not p.requires_grad:
                continue
            g, mod, _ = param_group(name)
            if g == "spec":
                self.average_into(name, [(c, n[c.id] if c.modality == mod else 0.0) for c in clients], clients)
            elif g in ("shared", "decoder"):
                self.average_into(name, [(c, bal[c.id]) for c in clients], clients)
        ups = [(c, c.local_prototype()) for c in clients]
        ups = [(c, P) for c, P in ups if P is not None]
        if ups:
            tot = sum(bal[c.id] for c, _ in ups)
            G = sum(P * (bal[c.id] / tot) for c, P in ups)
            for c in clients:
                c.global_protos = G.clone()
        print("  specialised adapters per modality, shared adapters + decoder + prototypes over all clients", flush=True)

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