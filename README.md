# FedSHA: Federated Medical Image Segmentation under Anatomy and Modality Heterogeneity via Hierarchical Adapters

This repository provides the official implementation of **FedSHA**, a federated learning framework for medical image segmentation where clients differ in both **anatomy** (organ) and **imaging modality** (CT, MRI).

FedSHA shares one grid of adapters on a frozen MedSAM encoder. In every block a gate picks the top-k adapters inside a sliding window that moves from a shared region in the early blocks to an organ-specific region in the last blocks. The server aggregates each adapter with weights from the clients' gates.

The framework supports segmentation experiments on six clients (liver, kidney, spleen, brain; CT and MRI) and enables fair comparison with several federated learning baselines.

---

## Dataset Preparation

The `dataset/` directory contains the script that generates the client-wise data splits (train / val / test by patient).

### Clients

| Client | Dataset | Modality | Organ  |
|--------|---------|----------|--------|
| 0      | LiTS17  | CT       | liver  |
| 1      | DLDS    | MRI      | liver  |
| 2      | KiTS23  | CT       | kidney |
| 3      | AMOS22  | MRI      | spleen |
| 4      | BraTS   | MRI      | brain  |
| 5      | BHSD    | CT       | brain  |

### Generating the Dataset

From the `dataset/` directory, run:

```bash
python3 generate_modorgan.py
```

The data is written to `dataset/modorgan6/{train,val,test}0/<client>.npz`.

Download the MedSAM ViT-B checkpoint `medsam_vit_b.pth` from the [MedSAM repository](https://github.com/bowang-lab/MedSAM) and place it in `system/`.

---

## Running Experiments

All federated learning experiments are executed from the `system/` directory using the provided shell script.

### Main Components

- `main.py` — Core federated learning training entry point
- `run.sh` — Script to launch experiments and tests
- `system/flcore/trainmodel/` — FedSHA and baseline networks
- `system/flcore/clients/`, `system/flcore/servers/` — Client and server implementations of all methods
- `system/utils/` — Helper utilities
- `wilcoxon.py`, `ablation_figs.py`, `qualitative_fig.py`, `comm_cost.py` — Statistical tests and figures

### Running Experiments

```bash
cd system
./run.sh <run> <gpu>
```

Example:

```bash
./run.sh fedsha 0
```

Hyperparameters for each method can be adjusted directly inside `run.sh`. Use `./run.sh list` to see all runs.

---

## Example Commands

### FedSHA (proposed)

```bash
nohup python3 main.py \
  -algo FedSHA \
  --arch fedsha \
  -mn fedsha \
  -data modorgan6 \
  -nc 6 \
  -gr 25 \
  -lbs 16 \
  -spe 2500 \
  --sam_ckpt ./medsam_vit_b.pth \
  -go fedsha \
  -gpu 0 \
  > logs_mo/fedsha.log 2>&1 &
```

### FedSHA-P (personalized)

Fine-tunes the trained FedSHA model on each client for 10 epochs.

```bash
nohup python3 main.py \
  -algo FedSHA \
  --arch fedsha \
  -mn fedsha \
  -data modorgan6 \
  -nc 6 \
  -lbs 16 \
  -spe 2500 \
  --sam_ckpt ./medsam_vit_b.pth \
  --ft_from fedsha \
  --ft_epochs 10 \
  --select client \
  -go ft \
  -gpu 0 \
  > logs_mo/ft.log 2>&1 &
```

### Testing

```bash
./run.sh test_all 0          # test the best checkpoints of all runs
python3 wilcoxon.py          # Wilcoxon signed-rank tests on per-case Dice
```

---

## Key Arguments

| Argument        | Description |
|-----------------|-------------|
| `-algo`         | Federated algorithm name (e.g., `FedSHA`, `FedMSA`, `FedMoSA`) |
| `--arch`        | Method / architecture (e.g., `fedsha`, `fedavg`, `fedmosa`, `fedmenu`) |
| `-mn`           | Model name (checkpoint prefix) |
| `-nc`           | Number of clients |
| `-data`         | Dataset to use (`modorgan6`) |
| `-gr`           | Number of global rounds |
| `-spe`          | Training slices per local epoch |
| `--grid`        | Adapter grid size (default 4, i.e. 4×4) |
| `--top_k`       | Adapters selected per token (default 4) |
| `--hard_window` | 0 = no sliding window (ablation) |
| `--win_stages`  | Fixed windows instead of the sliding window (`0,12` = organ region in all blocks) |
| `--win_side`    | Side length of the sliding window in adapters (default 2; e.g. `1.5` = smaller window) |
| `--expert_agg`  | Adapter aggregation: `gate` (default), `window`, `fedavg` |
| `--ft_from`     | FedSHA-P: fine-tune the FedSHA model of this run |

---

## Supported Methods

- **FedSHA** (proposed) — Shared adapter grid with sliding window and gate-informed aggregation
- **FedSHA-P** (proposed) — Personalized FedSHA with local fine-tuning
- **MSA-local / MSA-central** — Medical SAM adapter trained locally or on pooled data
- **FedMSA** — Federated learning with medical SAM adapter
- **FedProx**, **MOON** — Global federated baselines
- **Fed-MENU** — Federated multi-encoder network
- **FedMoSA** — Federated learning with specialized and shared adapters
- **Ditto**, **FedRep**, **FedHCA2**, **FedALA**, **FedDBE** — Personalized federated baselines

---

## Results

Experiment outputs are saved to the `results/` directory and checkpoints to `checkpoints/`. Training logs are written to `.log` files in `system/logs_mo/` for later inspection.

---

## Notes

- Experiments are executed asynchronously using `nohup`.
- Training logs are saved to `.log` files (e.g., `logs_mo/fedsha.log`).
- GPU selection is controlled via the `gpu` argument of `run.sh` (or `-gpu` in `main.py`).
