import random
import warnings

import numpy as np
import torch
import argparse

from flcore.trainmodel.models import FedSHA, MSAModel, MoSAModel, MENUModel
from flcore.servers.serverFedSHA import FedSHAServer
from flcore.servers.serverLocal import FedLocal
from flcore.servers.serverFedAvg import FedAvgSeg
from flcore.servers.serverMoSA import FedMoSA
from flcore.servers.serverHCA2 import FedHCA2
from flcore.servers.serverMENU import FedMENU
from flcore.servers.serverCentral import FedCentral

warnings.simplefilter("ignore")


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(32)

BASELINES = ("local", "central", "fedavg", "fedprox", "moon", "ditto", "fedrep", "fedhca2", "fedala", "feddbe")


def run(args):
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: [{i + 1}th/{args.times}] =============", flush=True)
        from utils.data_utils import print_dataset_stats
        sel = [int(c) for c in str(args.clients or "").split(",") if c.strip() != ""]
        print_dataset_stats(args.dataset, args.num_clients, str(i), sel or None)
        print("Creating server and clients ...")
        if args.arch == "fedmenu":                      # Fed-MENU: one sub-encoder per organ
            args.model = MENUModel(checkpoint_path=args.sam_ckpt, n_organs=args.menu_organs,
                                   img_size=args.img_size).to(args.device)
        elif args.arch == "fedmosa":                      # FedMoSA: specialised + shared adapters
            args.model = MoSAModel(checkpoint_path=args.sam_ckpt, img_size=args.img_size,
                                   use_checkpoint=bool(args.use_checkpoint)).to(args.device)
        elif args.arch in BASELINES:                    # common base network: dense MSA adapters
            args.model = MSAModel(checkpoint_path=args.sam_ckpt, img_size=args.img_size,
                                  use_checkpoint=bool(args.use_checkpoint)).to(args.device)
            if args.arch == "feddbe":
                args.model.enable_dbe()
                args.model.to(args.device)
        else:                                           # FedSHA (ours)
            stages = tuple(int(v) for v in args.win_stages.split(',')) if args.win_stages else None
            args.model = FedSHA(checkpoint_path=args.sam_ckpt, grid=args.grid, top_k=args.top_k, win_stages=stages,
                                win_overlap=args.win_overlap, win_side=args.win_side, hard_window=bool(args.hard_window),
                                img_size=args.img_size, use_checkpoint=bool(args.use_checkpoint)).to(args.device)
        servers = {"fedsha": FedSHAServer, "local": FedLocal, "central": FedCentral, "fedavg": FedAvgSeg, "fedprox": FedAvgSeg, "ditto": FedAvgSeg, "fedrep": FedAvgSeg, "moon": FedAvgSeg, "fedala": FedAvgSeg, "feddbe": FedAvgSeg, "fedhca2": FedHCA2,
                   "fedmosa": FedMoSA, "fedmenu": FedMENU}
        server = servers[args.arch](args, i)
        args.model = None
        torch.cuda.empty_cache()
        if args.test_only:
            server.test_only()
        else:
            server.train()


def get_args():
    p = argparse.ArgumentParser()
    # general
    p.add_argument('-go', "--goal", type=str, default="experiment", help="Run name (checkpoints / results)")
    p.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument('-gpu', "--gpu_index", type=int, default=0)
    p.add_argument('-data', "--dataset", type=str, default="modorgan6")
    p.add_argument('-nc', "--num_clients", type=int, default=6)
    p.add_argument('-t', "--times", type=int, default=1, help="Number of dataset splits / repeats")
    p.add_argument('-pv', "--prev", type=int, default=0)
    p.add_argument('-pr', "--prev_round", type=int, default=0, help="Resume from this round")
    p.add_argument('-algo', "--algorithm", type=str, default="FedSHA")
    p.add_argument('-mn', "--model_name", type=str, default="fedsha", help="Checkpoint file prefix")
    p.add_argument("--arch", type=str, default="fedsha", choices=["fedsha", "fedmosa", "fedmenu"] + list(BASELINES),
                   help="fedsha = ours; fedmosa / fedmenu = own networks; the rest = baselines on the MSA base network")
    p.add_argument("--sam_ckpt", type=str, default="./medsam_vit_b.pth")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--use_checkpoint", type=int, default=0)
    p.add_argument("--seed", type=int, default=32, help="Training seed (32 = default run; others for repeats)")
    p.add_argument("--clients", type=str, default="", help="Local baseline: train only these clients, e.g. 0,3")
    p.add_argument("--dump_preds", type=int, default=0, help="with --test_only 1: save predictions on fixed test slices for the qualitative figure")
    p.add_argument("--test_only", type=int, default=0, help="1 = only test the saved best checkpoints of run -go")
    p.add_argument("--hist_dir", type=str, default="../results/")
    p.add_argument("--log_dir", type=str, default="../logs/")
    p.add_argument("--ckpt_dir", type=str, default="../checkpoints/")
    # training (identical for every method)
    p.add_argument('-lbs', "--batch_size", type=int, default=16)
    p.add_argument('-lr', "--local_learning_rate", type=float, default=1e-4, help="(logging only; per-group lrs in clientseg.LR_DEFAULT)")
    p.add_argument('-gr', "--global_rounds", type=int, default=25)
    p.add_argument('-ls', "--local_steps", type=int, default=2, help="Local epochs per round")
    p.add_argument('-spe', "--slices_per_epoch", type=int, default=2500,
                   help="Random train slices per local epoch (-1 = smallest client's train size)")
    p.add_argument("--warmup_rounds", type=int, default=1)
    p.add_argument('-lam_bce', "--lam_bce", type=float, default=20.0, help="Seg focal loss weight")
    p.add_argument('-lam_dice', "--lam_dice", type=float, default=1.0, help="Seg Dice loss weight")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--select", type=str, default="mean", choices=["client", "mean"],
                   help="Model selection: mean = ONE global model, the round with the best mean validation per-case Dice")
    p.add_argument("--shared_head", type=int, default=1, help="1 = one output layer for all clients (aggregated)")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_rounds", type=int, default=10)
    # ours
    p.add_argument("--grid", type=int, default=4, help="Shared adapters on a G x G grid (2 x 2 region per organ)")
    p.add_argument("--top_k", type=int, default=4, help="Adapters the gate keeps per token")
    p.add_argument("--win_stages", type=str, default="",
                   help="s1,s2: fixed windows instead of the sliding window (0,12 = organ region in every block)")
    p.add_argument("--win_overlap", type=float, default=1/3,
                   help="An adapter is in the window if the window covers >= this fraction of it per axis")
    p.add_argument("--win_side", type=float, default=2.0,
                   help="Side length of the sliding window in adapters (2 = the 2 x 2 window; e.g. 1.5 = smaller)")
    p.add_argument("--hard_window", type=int, default=1, help="0 = no window, top-k of the whole grid (ablation)")
    p.add_argument("--expert_agg", type=str, default="gate", choices=["gate", "window", "fedavg"],
                   help="gate: gate mass | window: n_c over the clients with e in their window | fedavg: n_c over all")
    p.add_argument("--ft_from", type=str, default="", help="FedSHA-P: fine-tune the FedSHA model of run <goal>")
    p.add_argument("--ft_epochs", type=int, default=10, help="FedSHA-P fine-tuning epochs (slices_per_epoch each)")
    # baselines
    p.add_argument("--prox_mu", type=float, default=0.01, help="FedProx")
    p.add_argument("--moon_mu", type=float, default=1.0, help="MOON contrastive loss weight")
    p.add_argument("--moon_tau", type=float, default=0.5, help="MOON temperature")
    p.add_argument("--ditto_lambda", type=float, default=0.1, help="Ditto")
    p.add_argument("--ala_eta", type=float, default=1.0, help="FedALA: learning rate of the aggregation weights")
    p.add_argument("--ala_rand_percent", type=float, default=0.8, help="FedALA: fraction of one local epoch used by ALA")
    p.add_argument("--ala_layers", type=str, default="decoder", choices=["decoder", "body"],
                   help="FedALA: parameters with learned aggregation weights (rest: overwritten by the global model)")
    p.add_argument("--ala_threshold", type=float, default=0.1, help="FedALA: loss-std threshold of the first ALA")
    p.add_argument("--ala_num_pre_loss", type=int, default=10, help="FedALA: losses in the std window of the first ALA")
    p.add_argument("--ala_max_epochs", type=int, default=5, help="FedALA: cap on the epochs of the first ALA")
    p.add_argument("--dbe_kappa", type=float, default=50.0, help="FedDBE: weight kappa of the mean regularization (MSE)")
    p.add_argument("--dbe_momentum", type=float, default=0.1, help="FedDBE: momentum of the running mean representation")
    p.add_argument("--hca2_ca_c", type=float, default=0.4, help="FedHCA2 conflict-averse c (official)")
    p.add_argument("--hca2_alpha_init", type=float, default=0.1, help="FedHCA2 encoder hyperweight init (official)")
    p.add_argument("--hca2_beta_init", type=float, default=0.1, help="FedHCA2 decoder hyperweight init (official)")
    p.add_argument("--hca2_lr", type=float, default=0.01, help="FedHCA2 hyperweight SGD lr (official)")
    p.add_argument("--rep_head_epochs", type=int, default=1, help="FedRep: head-only epochs per round (rest: body)")
    p.add_argument("--menu_organs", type=int, default=3, help="Fed-MENU: number of organ sub-encoders")
    p.add_argument("--lam_menu_aux", type=float, default=1.0, help="Fed-MENU: AGD auxiliary loss weight")
    p.add_argument("--lam_proto_mosa", type=float, default=0.01, help="FedMoSA prototype alignment weight")
    args = p.parse_args()
    if args.seed != 32:
        set_seed(args.seed)
        import utils.data_utils as _du
        _du.RUN_SEED = args.seed
    return args


if __name__ == "__main__":
    args = get_args()
    import flcore.trainmodel.models as _models
    _models.SHARED_HEAD = bool(args.shared_head)            # before any client / server is built
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    else:
        args.device = "cpu"
    print("=" * 50)
    print(f"Algorithm: {args.algorithm} ({args.arch}) | backbone MedSAM ViT-B ({args.sam_ckpt})")
    print(f"Clients: {args.num_clients} | dataset: {args.dataset} | rounds: {args.global_rounds} | "
          f"local epochs: {args.local_steps} x {args.slices_per_epoch} slices | batch {args.batch_size}")
    if args.arch == "fedsha":
        print(f"Grid {args.grid}x{args.grid}, top-{args.top_k}, window {bool(args.hard_window)} "
              f"({args.win_stages or 'sliding'}), adapter aggregation {args.expert_agg}")
    print(f"Device: {args.device}")
    print("=" * 50)
    run(args)
