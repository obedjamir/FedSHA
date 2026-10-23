"""
Centralized training (upper reference): the data of all clients are pooled and ONE model is trained.
Body (MSA adapters + decoder trunk) is a single set of parameters shared by all datasets; every dataset keeps its
own mask slots (mask tokens + hypernetwork MLPs), since the classes differ (as for every other method).
Implementation: the clients' models share (tie) the same body parameters, and one AdamW optimiser trains the body and
all mask slots. Each round = the same number of slices as a federated round (every dataset: local_steps x
slices_per_epoch), with the batches of all datasets interleaved at random (proportional to their remaining batches).
Same loss, learning-rate schedule, evaluation and per-dataset model selection as every other method.
"""
import os
import copy
import random

import torch

from flcore.clients.clientFedAvg import clientFedAvg
from flcore.servers.serverbase import Server


def _tie(model, src, name):
    """Make `model`'s parameter `name` the very same Parameter object as in `src`."""
    mod_name, leaf = name.rsplit(".", 1)
    setattr(model.get_submodule(mod_name), leaf, src.get_parameter(name))


class FedCentral(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, {self.args.model_name}, num_clients:{args.num_clients}"
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp.replace(', ', '-')}-{args.goal}-{self.times}.h5")
        self.set_clients(args, clientFedAvg)
        self.global_model = None
        if getattr(args, "slices_per_epoch", 0) < 0:
            args.slices_per_epoch = min(c.train_size for c in self.clients)
        c0 = self.clients[0]
        self.body = c0.body_names()
        for c in self.clients[1:]:                                   # one body for all datasets
            for n in self.body:
                _tie(c.model, c0.model, n)
        named, seen = [], set()
        for c in self.clients:
            for n, p in c.model.named_parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    named.append((n, p))
        self.optimizer = c0.build_optimizer(named)
        self.rng = random.Random(getattr(args, "seed", 32))
        torch.cuda.empty_cache()
        print(f"Centralized | one body ({len(self.body)} tensors) shared by {len(self.clients)} datasets, "
              f"mask slots per dataset | {len(named)} trainable tensors")
        print("Finished creating server and clients.")

    def _ckpt(self, client, suffix=""):
        a = self.args
        return f"{a.ckpt_dir}/{a.dataset}-{a.model_name}-num_clients:{a.num_clients}-{client.id}-{a.goal}-{self.times}{suffix}.pth"

    def train_round(self, rnd):
        c0 = self.clients[0]
        f = c0.apply_lr(self.optimizer)
        for c in self.clients:
            c.model.train()
        tot, nb = {c.id: 0.0 for c in self.clients}, {c.id: 0 for c in self.clients}
        for epoch_idx in range(self.args.local_steps):
            iters = {c.id: iter(c.load_train_data(epoch=epoch_idx)) for c in self.clients}
            left = {c.id: len(c.train_data) // c.batch_size + (len(c.train_data) % c.batch_size > 0) for c in self.clients}
            by_id = {c.id: c for c in self.clients}
            while any(v > 0 for v in left.values()):
                ids = [i for i, v in left.items() if v > 0]
                cid = self.rng.choices(ids, weights=[left[i] for i in ids])[0]
                try:
                    x, y, _, _ = next(iters[cid])
                except StopIteration:
                    left[cid] = 0
                    continue
                left[cid] -= 1
                c = by_id[cid]
                x, y = c.prepare_batch(x, y)
                self.optimizer.zero_grad()
                loss = c.seg_loss(c.model(x), y)
                loss.backward()
                self.optimizer.step()
                tot[cid] += loss.item()
                nb[cid] += 1
        print(f"  lr x{f:.2f} | train loss: " + ", ".join(f"client {i}: {tot[i] / max(nb[i], 1):.4f}" for i in tot),
              flush=True)

    def train(self):
        if self.args.prev_round > 0:
            print("Loading Previous Checkpoint...")
            for c in self.clients:
                c.current_round = self.args.prev_round
                c.model.load_state_dict(torch.load(self._ckpt(c, "-ckpt"), map_location="cpu"))
        for i in range(self.global_rounds):
            if i < self.args.prev_round:
                continue
            print(f"\n------------- Round number: [{i + 1:3d}/{self.global_rounds}]-------------")
            print("==> Centralized training on the pooled data...", flush=True)
            for c in self.clients:
                c.current_round = i
            self.train_round(i)
            print("==> Evaluating models...", flush=True)
            mean_val, _ = self.evaluate()
            self.save_results(fn=self.hist_result_fn)
            self.select_checkpoints(mean_val, i, lambda c: self._ckpt(c))
            for c in self.clients:
                torch.save(c.model.state_dict(), self._ckpt(c, "-ckpt"))
            if self.should_stop(i):
                break
        print("==> Test with each dataset's best checkpoint...", flush=True)
        for c in self.clients:                                     # untie before loading per-dataset checkpoints
            c.model = copy.deepcopy(c.model)
            c.model.load_state_dict(torch.load(self._ckpt(c), map_location="cpu"))
        self.evaluate(val=False)
        self.save_results(fn=self.hist_result_fn)
