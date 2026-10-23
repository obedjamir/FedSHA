import copy
import json
import logging
import os
import random

import h5py
import numpy as np
import torch

class Server(object):
    def __init__(self, args, times):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.global_model = copy.deepcopy(args.model)

        self.num_clients = args.num_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.best_mean_test_acc = -1.0
        self.clients = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_loss = []
        self.rs_train_loss = []
        self.clients_test_accs = []
        self.clients_test_aucs = []
        self.clients_test_loss = []

        self.times = times

        self.set_seed(32)
        self.set_path(args)

        dir_alpha = 0.3

        self.actual_dataset = f"{self.dataset}-{self.num_clients}clients_alpha{dir_alpha:.1f}"
        logger_fn = os.path.join(args.log_dir, f"{args.algorithm}-{self.actual_dataset}.log")
        self.set_logger(save=True, fn=logger_fn)

        self.non_improve_rounds = 0

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def set_logger(self, save=False, fn=None):
        if save:
            fn = "testlog.log" if fn == None else fn
            logging.basicConfig(
                filename=fn,
                filemode="a",
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                level=logging.DEBUG
            )
        else:
            logging.basicConfig(
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                level=logging.DEBUG
            )

    def set_path(self, args):
        self.hist_dir = args.hist_dir
        self.log_dir = args.log_dir
        self.ckpt_dir = args.ckpt_dir
        if not os.path.exists(args.hist_dir):
            os.makedirs(args.hist_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    def set_clients(self, args, clientObj):
        dataset_id = str(self.times)

        for i in range(self.num_clients):
            client = clientObj(args, id=i, dataset_id=dataset_id, shared_model=self.global_model)
            self.clients.append(client)
        self.global_model = None                  # aggregates are written into the clients

    def _ckpt(self, client, suffix=""):
        a = self.args
        return f"{a.ckpt_dir}/{a.dataset}-{a.model_name}-num_clients:{a.num_clients}-{client.id}-{a.goal}-{self.times}{suffix}.pth"

    @torch.no_grad()
    def average_into(self, name, contrib, targets, row=None):
        """Weighted average of parameter `name` over the contributing clients, written into the target clients.
        contrib: [(client, weight)] (weights > 0, normalised here); targets: clients that receive the average.
        row: average only this row of the parameter (one mask token). Returns False if nobody contributes."""
        contrib = [(c, float(w)) for c, w in contrib if w > 0]
        if not contrib or not targets:
            return False
        tot = sum(w for _, w in contrib)
        avg = None
        for c, w in contrib:
            v = dict(c.model.named_parameters())[name].detach()
            v = (v if row is None else v[row]).float().cpu()
            avg = v * (w / tot) if avg is None else avg + v * (w / tot)
        for c in targets:
            p = dict(c.model.named_parameters())[name]
            if row is None:
                p.copy_(avg.to(p.device, p.dtype))
            else:
                p[row].copy_(avg.to(p.device, p.dtype))
        return True

    def select_checkpoints(self, mean_val, round_idx, best_path):
        """Model selection, identical for every method (--select):
          client (default): each client keeps the checkpoint of its own best validation round;
          mean            : one round for all clients, the one with the best mean validation score.
        Early stopping (should_stop) counts rounds in which no selected checkpoint improved."""
        if getattr(self.args, "select", "client") == "mean":
            if mean_val >= self.best_mean_test_acc:
                print(f"New best mean validation per-case Dice: {mean_val * 100:.2f}% (Round {round_idx + 1}) "
                      f"-> global model saved", flush=True)
                self.non_improve_rounds = 0
                for client in self.clients:
                    torch.save(client.model.state_dict(), best_path(client))
            else:
                self.non_improve_rounds += 1
                print(f"No improvement for {self.non_improve_rounds} consecutive round(s).", flush=True)
            return
        if not hasattr(self, "client_best"):
            self.client_best = {}                       # client id -> (best val score, round)
        scores = self.clients_test_accs[-1]
        improved = []
        for client, score in zip(self.clients, scores):
            if score > self.client_best.get(client.id, (-1.0, 0))[0]:
                self.client_best[client.id] = (score, round_idx + 1)
                torch.save(client.model.state_dict(), best_path(client))
                improved.append(client.id)
        best_mean = np.mean([v[0] for v in self.client_best.values()])
        print("Best per client: " + " | ".join(f"client {k}: {v[0] * 100:.2f}% (r{v[1]})"
                                               for k, v in sorted(self.client_best.items()))
              + f" | mean of bests: {best_mean * 100:.2f}%", flush=True)
        if improved:
            self.non_improve_rounds = 0
            print(f"Improved this round: clients {improved}", flush=True)
        else:
            self.non_improve_rounds += 1
            print(f"No client improved for {self.non_improve_rounds} consecutive round(s).", flush=True)

    def should_stop(self, round_idx):
        """Early stopping (all methods): stop once the mean validation score has not improved for
        `patience` consecutive rounds, checked only from round `min_rounds` on."""
        p, m = getattr(self.args, "patience", 5), getattr(self.args, "min_rounds", 10)
        if p > 0 and round_idx + 1 >= m and self.non_improve_rounds >= p:
            print(f"==> Early stop at round {round_idx + 1}: no improvement for {self.non_improve_rounds} rounds", flush=True)
            return True
        return False

    def save_results(self, fn=None, reset=True):
        if fn is None:
            algo = self.dataset + "_" + self.algorithm
            result_path = self.hist_dir

        if len(self.rs_test_acc):
            if fn is None:
                algo = algo + "_" + self.goal + "_" + str(self.times+1)
                file_path = os.path.join(result_path, "{}.h5".format(algo))
            else:
                file_path = fn
            print("File path: " + file_path)

            if not reset and os.path.exists(file_path):
                # Load existing history
                with h5py.File(file_path, 'r') as hf:
                    prev_rs_test_acc = hf['rs_test_acc'][:]
                    prev_rs_test_auc = hf['rs_test_auc'][:]
                    prev_rs_test_loss = hf['rs_test_loss'][:]
                    prev_clients_test_accs = hf['clients_test_accs'][:]
                    prev_clients_test_aucs = hf['clients_test_aucs'][:]
                    prev_clients_test_loss = hf['clients_test_loss'][:]

                # Append new history to old
                self.rs_test_acc = np.concatenate([prev_rs_test_acc, self.rs_test_acc]).tolist()
                self.rs_test_auc = np.concatenate([prev_rs_test_auc, self.rs_test_auc]).tolist()
                self.rs_test_loss = np.concatenate([prev_rs_test_loss, self.rs_test_loss]).tolist()
                self.clients_test_accs = np.concatenate([prev_clients_test_accs, self.clients_test_accs]).tolist()
                self.clients_test_aucs = np.concatenate([prev_clients_test_aucs, self.clients_test_aucs]).tolist()
                self.clients_test_loss = np.concatenate([prev_clients_test_loss, self.clients_test_loss]).tolist()
                print("Previous History Loaded...")

            # Save (overwrite with full history)
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
                hf.create_dataset('clients_test_accs', data=self.clients_test_accs)
                hf.create_dataset('clients_test_aucs', data=self.clients_test_aucs)
                hf.create_dataset('clients_test_loss', data=self.clients_test_loss)

    def test_metrics(self, temp_model=None, val=True):
        """ A personalized evaluation scheme (test_acc's do not average based on num_samples) """
        test_accs, test_aucs, test_losses, test_nums = [], [], [], []
        ids = [c.id for c in self.clients]
        filtered_ids = ids
        for c in self.clients:
            if c.id in filtered_ids:
                test_acc, test_auc, test_loss, test_num = c.test_metrics(temp_model, val=val)
                test_accs.append(test_acc)
                test_aucs.append(test_auc)
                test_losses.append(test_loss)
                test_nums.append(test_num)
        return ids, test_accs, test_aucs, test_losses, test_nums

    def evaluate(self, temp_model=None, mode="personalized", val=True):
        ids, test_accs, test_aucs, test_losses, test_nums = self.test_metrics(temp_model, val=val)
        self.clients_test_accs.append(copy.deepcopy(test_accs))
        self.clients_test_aucs.append(copy.deepcopy(test_aucs))
        self.clients_test_loss.append(copy.deepcopy(test_losses))
        if mode == "personalized":
            mean_test_acc, mean_test_auc, mean_test_loss = np.mean(test_accs), np.mean(test_aucs), np.mean(test_losses)
        elif mode == "global":
            mean_test_acc, mean_test_auc, mean_test_loss = np.average(test_accs, weights=test_nums), np.average(test_aucs, weights=test_nums), np.average(test_losses, weights=test_nums)
        else:
            raise NotImplementedError

        print(test_accs)
        print(test_aucs)
        self.best_mean_test_acc = max(mean_test_acc, self.best_mean_test_acc)
        self.rs_test_acc.append(mean_test_acc)
        self.rs_test_auc.append(mean_test_auc)
        self.rs_test_loss.append(mean_test_loss)
        if val:
            print(f"==> val_loss: {mean_test_loss:.5f} | mean_val_score: {mean_test_acc*100:.2f}% | best_score: {self.best_mean_test_acc*100:.2f}%\n")
        else:
            print(f"==> test_loss: {mean_test_loss:.5f} | mean_test_score: {mean_test_acc*100:.2f}% | best_score: {self.best_mean_test_acc*100:.2f}%\n")
        return mean_test_acc, self.best_mean_test_acc

    def test_only(self):
        """Evaluate every client's saved best checkpoint (run -go <goal>) on its test set, without training."""
        print(f"\n========== TEST ONLY: {self.args.algorithm} ({self.args.arch}) | run '{self.args.goal}' ==========",
              flush=True)
        for c in self.clients:
            c.model = copy.deepcopy(c.model)                    # independent copy (centralized ties the body)
            c.model.load_state_dict(torch.load(self._ckpt(c), map_location="cpu"))
            c.model.to(self.args.device)
        if getattr(self.args, "dump_preds", 0):
            self.dump_predictions()
            return
        self.evaluate(val=False)
        cdir = os.path.join(getattr(self.args, "hist_dir", "../results"), "cases")
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, f"{self.args.goal}.json"), "w") as fo:          # per-case Dice (Wilcoxon tests)
            json.dump({str(c.id): getattr(c, "case_scores", {}) for c in self.clients}, fo)
        keys = ["dice", "iou", "dice_global", "dice_case", "dice_case_client_mean", "iou_case_client_mean"]
        rows = [(c.id, c.dataset, getattr(c, "task_metrics", {})) for c in self.clients]
        print(f"SUMMARY {self.args.algorithm} ({self.args.goal}) | " + " | ".join(
            f"client {i}: " + ", ".join(f"{k} {m.get(k, 0) * 100:.2f}" for k in keys) for i, _, m in rows))
        print(f"SUMMARY {self.args.algorithm} ({self.args.goal}) | per-case Dice mean +- std over cases: " + " | ".join(
            f"client {i}: {m.get('dice_case_client_mean', 0) * 100:.2f} +- {m.get('dice_case_client_std', 0) * 100:.2f}"
            for i, _, m in rows))
        print(f"SUMMARY {self.args.algorithm} ({self.args.goal}) | per-case IoU mean +- std over cases: " + " | ".join(
            f"client {i}: {m.get('iou_case_client_mean', 0) * 100:.2f} +- {m.get('iou_case_client_std', 0) * 100:.2f}"
            for i, _, m in rows))
        for k in keys:
            print(f"SUMMARY {self.args.algorithm} ({self.args.goal}) | mean {k}: "
                  f"{np.mean([m.get(k, 0) for _, _, m in rows]) * 100:.2f}", flush=True)


    @torch.no_grad()
    def dump_predictions(self, n_cases=20):
        """Qualitative results: for every client, the slice with the largest ground-truth area in each of the first
        n_cases test cases (the same slices for every method), saved with the prediction of this run to
        ../results/preds/<goal>.npz (image = centre slice, uint8; masks as bool)."""
        out_dir = os.path.join(getattr(self.args, "hist_dir", "../results"), "preds")
        os.makedirs(out_dir, exist_ok=True)
        dump = {}
        for c in self.clients:
            c.ensure_organ() if hasattr(c, "ensure_organ") else None
            model = c.get_eval_model()
            model.eval()
            best = {}                                                     # case -> (area, img, gt, pred)
            loader = c.load_test_data()
            for b, (x, y, _, cases) in enumerate(loader):
                x, yt = c.prepare_batch(x, y)
                c.on_batch(loader, b)
                prob = torch.sigmoid(c.seg_logits(model(x), yt))
                pred = (prob > 0.5)[:, 0].cpu().numpy()
                gt = (yt[:, 0] > 0.5).cpu().numpy()
                img = x[:, 1].float().cpu().numpy()
                for i, case in enumerate(cases):
                    a = int(gt[i].sum())
                    if a > 0 and a > best.get(case, (0,))[0]:
                        im = img[i]
                        im = ((im - im.min()) / max(float(im.max() - im.min()), 1e-6) * 255).astype(np.uint8)
                        best[case] = (a, im, gt[i].copy(), pred[i].copy())
            model.clear_routing() if hasattr(model, "clear_routing") else None
            for j, case in enumerate(sorted(best)[:n_cases]):
                _, im, g, p = best[case]
                k = f"c{c.id}_{j}"
                dump[k + "_img"], dump[k + "_gt"], dump[k + "_pred"] = im, g, p
                dump[k + "_case"] = np.array(str(case))
            print(f"Client {c.id}: {min(len(best), n_cases)} slices dumped", flush=True)
        fn = os.path.join(out_dir, f"{self.args.goal}.npz")
        np.savez_compressed(fn, **dump)
        print(f"predictions saved to {fn}", flush=True)
