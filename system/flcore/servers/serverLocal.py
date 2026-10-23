"""
Local baseline: every client trains alone, no communication, on the common base network (FedHCA2's:
frozen SAM + dense adapters + per-task SAM decoders + thin heads (CenterNet for det)). Same data, epochs per round, rounds,
optimiser and learning rates as every other method.

Model selection (--select, the same for every method):
  mean            : ONE round for all clients: the round with the best MEAN
                    validation score; early stopping on the mean (patience / min_rounds as everywhere)
  client          : each client keeps its own best round; training stops when no client improved for
                    --patience rounds (default; the same rule as the federated methods)
--clients "0,3" trains only those clients (one process per client lets learning rates be swept in
parallel); the per-epoch slice budget still follows the smallest client of the whole federation.
"""
import os
import json
import torch

from flcore.clients.clientLocal import clientLocal
from flcore.servers.serverbase import Server
from utils.data_utils import load_split


class FedLocal(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        ids = [int(i) for i in str(getattr(args, "clients", "") or "").split(",") if i.strip() != ""]
        self.client_ids = ids or list(range(args.num_clients))

        # slice budget from the whole federation (identical to the federated runs)
        if getattr(args, "slices_per_epoch", 0) < 0:
            sizes = [len(load_split(args.dataset, i, str(times), "train")) for i in range(args.num_clients)]
            args.slices_per_epoch = min(sizes)
        print(f"Local baseline | common base network | clients {self.client_ids}"
              f" | lr scale {getattr(args, 'lr_scale', 1.0)}"
              f" | per epoch: {args.slices_per_epoch}")

        self.clients = [clientLocal(args, id=i, dataset_id=str(self.times), shared_model=self.global_model)
                        for i in self.client_ids]
        self.global_model = None                                  # not needed: no aggregation
        torch.cuda.empty_cache()
        self.best = {c.id: -1.0 for c in self.clients}
        self.best_round = {c.id: 0 for c in self.clients}
        self.history = {c.id: [] for c in self.clients}
        self.no_improve = {c.id: 0 for c in self.clients}
        self.stopped = set()
        self.select = getattr(args, "select", "client")          # same rule as every federated method
        self.best_mean, self.best_mean_round, self.non_improve_rounds = -1.0, 0, 0
        print(f"Model selection: {self.select}")
        print("Finished creating clients.")

    def _ckpt(self, client, suffix=""):
        return (f"{self.args.ckpt_dir}/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}"
                f"-{client.id}-{self.args.goal}-{self.times}{suffix}.pth")

    def train(self):
        if self.args.prev_round > 0:
            print("Loading Previous Checkpoint...")
            for c in self.clients:
                c.current_round = self.args.prev_round
                c.ensure_organ()
                c.model.load_state_dict(torch.load(self._ckpt(c, "-ckpt"), map_location="cpu"))
                c.model.to(self.args.device)

        for i in range(self.global_rounds):
            if i < self.args.prev_round:
                continue
            print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------", flush=True)
            active = [c for c in self.clients if c.id not in self.stopped]
            for c in active:
                c.current_round = i
                c.train()

            print("==> Evaluating local models...", flush=True)
            p, m = getattr(self.args, "patience", 5), getattr(self.args, "min_rounds", 10)
            if self.select == "mean":
                scores = {}
                for c in active:
                    score, _, loss, _ = c.test_metrics(val=True)
                    scores[c.id] = score
                    self.history[c.id].append({"round": i + 1, "score": score, **c.task_metrics})
                    torch.save(c.model.state_dict(), self._ckpt(c, "-ckpt"))
                mean = sum(scores.values()) / len(scores)
                print(" | ".join(f"client {k}: {v * 100:.2f}%" for k, v in scores.items()), flush=True)
                if mean > self.best_mean:
                    self.best_mean, self.best_mean_round, self.non_improve_rounds = mean, i + 1, 0
                    for c in active:
                        torch.save(c.model.state_dict(), self._ckpt(c))
                        self.best[c.id], self.best_round[c.id] = scores[c.id], i + 1
                else:
                    self.non_improve_rounds += 1
                print(f"==> mean_val_score: {mean * 100:.2f}% | best: {self.best_mean * 100:.2f}% "
                      f"(r{self.best_mean_round})", flush=True)
                if p > 0 and i + 1 >= m and self.non_improve_rounds >= p:
                    print(f"==> Early stop at round {i + 1}: no improvement for {p} rounds", flush=True)
                    break
                continue
            # each client keeps its own best round; all clients stop together when none improved for p rounds
            # (the same rule as the federated methods)
            improved, scores = [], {}
            for c in active:
                score, _, loss, _ = c.test_metrics(val=True)
                scores[c.id] = score
                self.history[c.id].append({"round": i + 1, "score": score, **c.task_metrics})
                if score > self.best[c.id]:
                    self.best[c.id], self.best_round[c.id] = score, i + 1
                    torch.save(c.model.state_dict(), self._ckpt(c))
                    improved.append(c.id)
                torch.save(c.model.state_dict(), self._ckpt(c, "-ckpt"))
            print(" | ".join(f"client {k}: {v * 100:.2f}%" for k, v in scores.items()), flush=True)
            print(f"==> mean_val_score: {100 * sum(scores.values()) / len(scores):.2f}%", flush=True)
            print("Best per client: " + " | ".join(f"client {k}: {v * 100:.2f}% (r{self.best_round[k]})"
                                                   for k, v in self.best.items())
                  + f" | mean of bests: {100 * sum(self.best.values()) / len(self.best):.2f}%", flush=True)
            self.non_improve_rounds = 0 if improved else self.non_improve_rounds + 1
            if p > 0 and i + 1 >= m and self.non_improve_rounds >= p:
                print(f"==> Early stop at round {i + 1}: no client improved for {p} rounds", flush=True)
                break

        # ---- final test: the selected checkpoint(s) ----
        print("==> Test with the best-mean-round checkpoints..." if self.select == "mean"
              else "==> Test with each client's best checkpoint...", flush=True)
        final = {}
        for c in self.clients:
            c.ensure_organ()
            c.model.load_state_dict(torch.load(self._ckpt(c), map_location="cpu"))
            c.model.to(self.args.device)
            score, _, _, _ = c.test_metrics(val=False)
            final[c.id] = {"best_round": self.best_round[c.id], "val_score": self.best[c.id],
                           "test_score": score, **c.task_metrics}
        print(f"==> mean_test_score: {100 * sum(v['test_score'] for v in final.values()) / len(final):.2f}%", flush=True)
        os.makedirs(self.args.hist_dir, exist_ok=True)
        out = os.path.join(self.args.hist_dir, f"local-{self.args.goal}-{self.times}.json")
        with open(out, "w") as f:
            json.dump({"final": final, "history": self.history, "lrs": {c.id: c.lrs for c in self.clients}}, f, indent=1)
        print(f"Saved {out}")
