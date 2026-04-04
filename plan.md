# RF-DETR Playground — Implementation Plan

## Overview

Build a self-contained playground for training, evaluating, and benchmarking RF-DETR object detection models. The project ships with 3 open-source datasets, provides simple CLI commands for training/evaluation, generates rich metrics and visualizations, and can run locally or on an EC2 instance with minimal setup.

---

## Architecture

```
rf-detr-playground/
├── README.md                  # Quick-start guide
├── pyproject.toml             # Python packaging (pip install -e .)
├── Makefile                   # Convenience targets (train, eval, deploy, etc.)
├── configs/
│   ├── default.yaml           # Base training config (lr, epochs, batch size, etc.)
│   ├── coco_sample.yaml       # COCO 128-image sample config
│   ├── pascal_voc.yaml        # Pascal VOC config
│   └── aquarium.yaml          # Aquarium dataset config
├── scripts/
│   ├── download_datasets.py   # Downloads & converts all 3 datasets to COCO format
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Evaluation entry point (mAP, per-class metrics)
│   ├── predict.py             # Run inference on images/video and visualize
│   ├── benchmark.py           # Compare multiple models side-by-side
│   └── visualize.py           # Generate metric plots & detection overlays
├── src/
│   └── rf_detr_playground/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── downloader.py      # Dataset download & cache logic
│       │   ├── converter.py       # Format converters (VOC XML → COCO JSON)
│       │   └── registry.py        # Dataset registry (name → download/config)
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py         # Wraps rfdetr.RFDETRBase/Large .train() API
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py       # COCO-style mAP evaluation via pycocotools
│       │   └── metrics.py         # Metric computation helpers
│       └── visualization/
│           ├── __init__.py
│           ├── plots.py           # mAP curves, loss curves, per-class bar charts
│           └── detections.py      # Draw bounding boxes on images
├── deploy/
│   ├── setup_ec2.sh           # EC2 bootstrap script (install CUDA, drivers, deps)
│   ├── user_data.sh           # EC2 user-data for launch template
│   └── README.md              # Deployment instructions
├── notebooks/
│   └── exploration.ipynb      # Interactive notebook for experimentation
└── tests/
    ├── test_data.py
    ├── test_training.py
    └── test_evaluation.py
```

---

## Datasets (3 preloaded)

| # | Dataset | Source | Why |
|---|---------|--------|-----|
| 1 | **COCO 2017 (128-image sample)** | Ultralytics COCO128 mirror | Industry-standard benchmark; small sample for fast iteration |
| 2 | **Pascal VOC 2012** | Host mirror / HuggingFace | Classic 20-class benchmark; tests format conversion pipeline |
| 3 | **Aquarium** | Roboflow Universe (public) | Small, real-world, niche dataset (7 classes); great for fine-tuning demos |

All datasets are converted to **COCO JSON format** (the native format for RF-DETR) by the download script. The dataset registry makes it trivial to add more datasets later.

---

## Step-by-step Implementation

### Step 1: Project scaffolding & dependencies
- Create `pyproject.toml` with dependencies: `rfdetr`, `torch`, `torchvision`, `pycocotools`, `supervision`, `matplotlib`, `seaborn`, `pyyaml`, `click`, `tqdm`, `Pillow`
- Create the directory structure above
- Create `Makefile` with targets: `install`, `download-data`, `train`, `evaluate`, `predict`, `benchmark`
- Create `.gitignore` (data/, outputs/, *.pt, __pycache__, etc.)

### Step 2: Dataset pipeline
- Implement `src/rf_detr_playground/data/registry.py` — a dict mapping dataset names to download URLs, class lists, and conversion functions
- Implement `src/rf_detr_playground/data/downloader.py` — downloads and extracts datasets to `data/<dataset_name>/`, with caching (skip if already present)
- Implement `src/rf_detr_playground/data/converter.py` — converts Pascal VOC XML annotations to COCO JSON; COCO128 and Aquarium are already in COCO format or can be fetched in COCO format from Roboflow
- Implement `scripts/download_datasets.py` — CLI that calls the above for one or all datasets

### Step 3: Training wrapper
- Implement `src/rf_detr_playground/training/trainer.py`:
  - Wraps `rfdetr.RFDETRBase()` and `rfdetr.RFDETRLarge()` 
  - Reads a YAML config for hyperparams (epochs, lr, batch_size, grad_accum, img_size, etc.)
  - Calls `.train(dataset_dir=..., epochs=..., ...)` 
  - Saves checkpoints and training logs to `outputs/<run_name>/`
- Implement `scripts/train.py` — CLI entry point: `python scripts/train.py --config configs/aquarium.yaml --model base`
- Implement YAML config files for each dataset

### Step 4: Evaluation & metrics
- Implement `src/rf_detr_playground/evaluation/evaluator.py`:
  - Loads a trained model checkpoint
  - Runs inference on the validation set
  - Computes COCO metrics via `pycocotools` (mAP@50, mAP@50:95, mAP-small/medium/large)
  - Computes per-class AP
  - Returns structured results dict
- Implement `src/rf_detr_playground/evaluation/metrics.py` — helpers for precision-recall computation, confusion matrix generation
- Implement `scripts/evaluate.py` — CLI: `python scripts/evaluate.py --checkpoint outputs/run1/best.pt --dataset aquarium`

### Step 5: Visualization
- Implement `src/rf_detr_playground/visualization/plots.py`:
  - **mAP bar chart** — per-class AP as horizontal bars
  - **Precision-Recall curves** — per-class and aggregate
  - **Confusion matrix** — heatmap of predicted vs actual classes
  - **Training loss curve** — loss over epochs (parsed from training logs)
  - **Model comparison chart** — side-by-side mAP for multiple runs
- Implement `src/rf_detr_playground/visualization/detections.py`:
  - Draw bounding boxes with class labels and confidence scores on images
  - Uses `supervision` library for clean annotation rendering
  - Save annotated images to `outputs/<run>/visualizations/`
- Implement `scripts/predict.py` — run inference on images and save visualized results
- Implement `scripts/visualize.py` — generate all plots from evaluation results
- Implement `scripts/benchmark.py` — evaluate multiple checkpoints, generate comparison

### Step 6: EC2 deployment support
- Create `deploy/setup_ec2.sh`:
  - Installs NVIDIA drivers, CUDA toolkit
  - Installs Python 3.11+, pip
  - Clones the repo, installs dependencies
  - Downloads datasets
- Create `deploy/user_data.sh` — minimal user-data script for EC2 launch
- Create `deploy/README.md` — step-by-step instructions for:
  - Recommended instance types (g4dn.xlarge for budget, g5.xlarge for speed)
  - AMI selection (Deep Learning AMI)
  - Security group setup
  - SSH access and tmux usage for long training runs

### Step 7: Notebook & documentation
- Create `notebooks/exploration.ipynb` — interactive notebook showing full workflow: download data → train → evaluate → visualize
- Create `README.md` with:
  - Project overview and features
  - Quick-start (3 commands to first results)
  - Dataset descriptions
  - Configuration reference
  - EC2 deployment guide
  - Example outputs (placeholder for screenshots)

### Step 8: Tests
- `tests/test_data.py` — test dataset registry, converter logic (with small fixture data)
- `tests/test_training.py` — test config loading, trainer initialization (no actual GPU training)
- `tests/test_evaluation.py` — test metric computation with mock predictions

---

## Key Design Decisions

1. **COCO format as canonical** — RF-DETR natively uses COCO format; all datasets are converted to this on download, avoiding format issues during training
2. **YAML configs** — every training run is fully specified by a YAML file, making experiments reproducible and easy to tweak
3. **`supervision` library for visualization** — Roboflow's library provides high-quality bbox rendering out of the box
4. **Click CLI** — all scripts use Click for clean argument parsing and help text
5. **Makefile as orchestrator** — `make train CONFIG=configs/aquarium.yaml` is the simplest possible UX
6. **EC2 via shell scripts (not Terraform/CDK)** — keeps deployment simple and accessible; users just need AWS CLI

---

## Metrics & Visualizations Summary

| Metric / Viz | Description |
|---|---|
| mAP@50 | Mean Average Precision at IoU 0.50 |
| mAP@50:95 | Mean AP averaged over IoU thresholds 0.50–0.95 |
| mAP-S/M/L | AP broken down by object size |
| Per-class AP | Bar chart of AP for each class |
| Precision-Recall curves | Per-class and aggregated |
| Confusion matrix | Heatmap showing misclassifications |
| Training loss curve | Loss over epochs |
| Detection overlays | Side-by-side ground truth vs predictions on sample images |
| Model comparison | Table + chart comparing Base vs Large across datasets |
