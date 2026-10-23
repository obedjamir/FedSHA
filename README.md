# FedSHA

Federated segmentation across organs and modalities with a shared adapter grid on a frozen MedSAM encoder.

Each client holds one organ in one modality (CT or MRI). All clients share one 4×4 grid of adapters. In every encoder block a gate picks the top-4 adapters inside a sliding window. The window starts at the grid centre in the early blocks, which all clients share, and slides to the client's own organ region in the last blocks. The server averages each adapter with weights from how much the clients' gates actually used it.

## Method

- **Backbone.** MedSAM ViT-B. The image encoder is frozen. The adapters, gates, scale/shift (γ, β) and the mask decoder are trained.
- **Adapter grid.** One shared 4×4 grid, used by all 12 blocks. The server sorts the organ names and gives organ *i* the 2×2 region *i*.
- **Sliding window.** The window centre moves from the grid centre μ₀ to the organ centre μₒ:
  μ_l = (1 − τ) μ₀ + τ μₒ, with τ = (l − 1)/(L − 1).
  An adapter is in the window if the window covers at least 1/3 of it on each axis. Blocks 1–4 use the centre region, blocks 5–8 a 3×3 window and blocks 9–12 the organ region.
- **Gate.** Per block and branch (attention, MLP), a top-4 softmax gate over the adapters in the window, followed by a per-block scale and shift.
- **Gate-informed aggregation.**
  - Adapter *e* is averaged over the clients with weights equal to its gate mass. Gate mass is the mean gate probability over tokens and blocks, and it is 0 outside the window.
  - Gate rows use the same weights as their adapters.
  - γ/β and the mask decoder are averaged over all clients with weights n_c (training volumes).
  - Mask tokens stay local.
- **FedSHA-P.** Each client fine-tunes the FedSHA model on its own data for 10 epochs and keeps its best validation epoch.

## Results

Test Dice (%) on the six-client federation. The higher score in each pair is in bold.

| Method | LiTS17 (CT) | DLDS (MRI) | KiTS23 (CT) | AMOS22 (MRI) | BraTS (MRI) | BHSD (CT) | Mean |
|---|---|---|---|---|---|---|---|
| FedSHA | 93.52 | 92.24 | 90.26 | 93.48 | 85.66 | 45.74 | 83.48 |
| FedSHA-P | 94.65 | 92.32 | 91.82 | 94.03 | 86.45 | 48.47 | 84.63 |

The baselines are FedMSA (FedAvg), FedProx, MOON, Fed-MENU, FedMoSA, Ditto, FedRep, FedHCA2, FedALA, FedDBE, MSA-local and MSA-central. They all use the same backbone and training budget. The full comparison and the Wilcoxon tests are in the paper.

Ablations (mean test Dice):

| Variant | Mean |
|---|---|
| FedSHA | 83.48 |
| w/o gate-informed aggregation | 82.91 |
| organ region in all blocks | 83.00 |
| w/o window | 65.42 |

## Repository layout

```
system/
├── main.py                  # entry point (arguments, model and server)
├── run.sh                   # all runs, tests and helpers
├── flcore/
│   ├── trainmodel/models.py # FedSHA, SlidingWindow, baseline networks
│   ├── clients/             # clientseg (shared training / evaluation), clientFedSHA, baselines
│   └── servers/             # serverFedSHA, serverbase, baselines
├── utils/data_utils.py      # client splits and per-epoch slice sampling
├── wilcoxon.py              # paired Wilcoxon tests on per-case Dice
├── ablation_figs.py         # ablation figure
├── qualitative_fig.py       # qualitative figure
└── comm_cost.py             # parameters sent per client
```

## Setup

```bash
pip install torch torchvision numpy scipy h5py matplotlib pillow
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download the MedSAM ViT-B checkpoint `medsam_vit_b.pth` from the [MedSAM repository](https://github.com/bowang-lab/MedSAM) and place it in `system/`.

### Data

Clients are read from `dataset/modorgan6/{train,val,test}0/<client>.npz`. Each file lists the slice PNGs and their masks, plus a `meta` entry with the modality, organ and segmentation regions. Splits are by patient.

| Client | Dataset | Modality | Organ |
|---|---|---|---|
| 0 | LiTS17 | CT | liver |
| 1 | DLDS | MRI | liver |
| 2 | KiTS23 | CT | kidney |
| 3 | AMOS22 | MRI | spleen |
| 4 | BraTS | MRI | brain |
| 5 | BHSD | CT | brain |

## Usage

Run from `system/`. Each run writes `logs_mo/<run>.log`, its checkpoints to `../checkpoints/` and its results to `../results/`.

```bash
./run.sh fedsha 0              # FedSHA on GPU 0
./run.sh ft 0                  # FedSHA-P (fine-tunes the fedsha run; FT_FROM=<run> to change)

./run.sh no_window 0           # ablations
./run.sh no_gate 1
./run.sh organ_only 1

./run.sh baselines "0 1 2"     # global baselines, two runs per GPU
./run.sh personal "0 1 2"      # personalized baselines
./run.sh list                  # all run names

./run.sh test_all 0            # test the best checkpoints of all runs -> logs_mo/test_all.log
./run.sh dump_all 0            # save predictions for the qualitative figure
python3 wilcoxon.py            # Wilcoxon tests (needs test_all first)
python3 ablation_figs.py
```

Main FedSHA options in `main.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--grid` | 4 | adapter grid size (G×G) |
| `--top_k` | 4 | adapters per token |
| `--win_overlap` | 1/3 | coverage needed for an adapter to be in the window |
| `--hard_window` | 1 | 0 = no window, top-k over the whole grid |
| `--win_stages` | sliding | `s1,s2` for fixed windows (`0,12` = organ region in all blocks) |
| `--expert_agg` | gate | `gate`, `window` (n_c over clients with the adapter in their window) or `fedavg` |
| `--ft_from`, `--ft_epochs` | –, 10 | FedSHA-P |

Training settings (the same for every method): 25 rounds, 2 local epochs of 2,500 slices, batch 16, 512×512 input, early stopping with patience 5 after round 10.

### Checkpoints from the older code

Checkpoints from the earlier version of this code use the model name `atmoe` and the run names `gate_eq`, `no_usage_proto` and `feddbe_alt`. Run this once to rename them:

```bash
./run.sh migrate
```

## Citation

```bibtex
@inproceedings{fedsha,
  title     = {FedSHA: ...},
  author    = {...},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026}
}
```

## Acknowledgements

Built on [Segment Anything](https://github.com/facebookresearch/segment-anything), [MedSAM](https://github.com/bowang-lab/MedSAM) and [PFLlib](https://github.com/TsingZ0/PFLlib).
