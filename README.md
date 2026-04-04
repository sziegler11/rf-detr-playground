# RF-DETR Playground

A self-contained playground for training, evaluating, and benchmarking [RF-DETR](https://github.com/roboflow/rf-detr) object detection models. Ships with 3 open-source datasets, simple CLI commands, rich evaluation metrics, and EC2 deployment support.

## Features

- **3 preloaded datasets** — COCO128 (sample), Pascal VOC 2012, Aquarium
- **Simple CLI** — train, evaluate, predict, and benchmark with one command
- **Rich evaluation** — mAP@50:95, per-class AP, precision-recall curves, confusion matrices
- **Visualizations** — detection overlays, training curves, model comparison charts
- **EC2-ready** — bootstrap scripts for GPU instances with one command
- **Configurable** — YAML configs for reproducible experiments

## Quick Start

```bash
# 1. Install
pip install -e ".[all]"

# 2. Download datasets
make download-data

# 3. Train a model
make train CONFIG=configs/aquarium.yaml

# 4. Evaluate
make evaluate CHECKPOINT=outputs/latest/best.pt DATASET=aquarium
```

## Project Structure

```
rf-detr-playground/
├── configs/              # Training YAML configs
│   ├── aquarium.yaml
│   ├── coco_sample.yaml
│   └── pascal_voc.yaml
├── scripts/              # CLI entry points
│   ├── download_datasets.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── benchmark.py
│   └── visualize.py
├── src/rf_detr_playground/
│   ├── data/             # Dataset download, conversion, registry
│   ├── training/         # Training wrapper
│   ├── evaluation/       # COCO metrics, per-class AP
│   └── visualization/    # Plots and detection overlays
├── deploy/               # EC2 setup scripts
├── notebooks/            # Interactive exploration
└── tests/                # Unit tests
```

## Datasets

| Dataset | Classes | Images | Description |
|---------|---------|--------|-------------|
| **COCO128** | 80 | 128 | COCO sample for fast iteration |
| **Pascal VOC 2012** | 20 | ~17k | Classic detection benchmark |
| **Aquarium** | 7 | ~640 | Underwater object detection |

Download individually or all at once:

```bash
python scripts/download_datasets.py --dataset aquarium
python scripts/download_datasets.py --all
```

The Aquarium dataset requires a free Roboflow API key:
```bash
export ROBOFLOW_API_KEY=your_key_here
```

## Training

Train with a config file:

```bash
# Default (Aquarium, RF-DETR Base, 30 epochs)
python scripts/train.py --config configs/aquarium.yaml

# Override parameters
python scripts/train.py --config configs/aquarium.yaml --model large --epochs 50 --lr 2e-4

# Quick sanity check with COCO128
python scripts/train.py --config configs/coco_sample.yaml --epochs 5
```

Or use `make`:

```bash
make train CONFIG=configs/aquarium.yaml MODEL=base
```

### Model Sizes

| Size | Params | COCO AP | Speed (T4) |
|------|--------|---------|------------|
| Nano | 30.5M | 48.4 | 2.3ms |
| Small | 32.1M | 53.0 | 3.5ms |
| Medium | 33.7M | 54.7 | 4.4ms |
| Base | 33.9M | 56.5 | 6.8ms |
| Large | 126.4M | 58.6 | 11.5ms |

## Evaluation

```bash
# Evaluate a checkpoint
python scripts/evaluate.py \
    --checkpoint outputs/latest/best.pt \
    --dataset aquarium \
    --model base

# Compare multiple models
python scripts/benchmark.py -m base -m large -d aquarium
```

### Metrics Generated

- **mAP@50:95** — primary COCO metric
- **mAP@50, mAP@75** — at specific IoU thresholds
- **mAP by size** — small, medium, large objects
- **Per-class AP** — bar chart for each class
- **Inference speed** — average ms per image

## Inference

```bash
# Run on images
python scripts/predict.py \
    --images data/aquarium/valid \
    --model base \
    --confidence 0.5

# With a fine-tuned checkpoint
python scripts/predict.py \
    --checkpoint outputs/latest/best.pt \
    --images path/to/images \
    --dataset aquarium
```

## Configuration

Training is configured via YAML files in `configs/`. Key parameters:

```yaml
dataset_name: aquarium       # Dataset to use
model_size: base             # nano/small/medium/base/large
epochs: 30                   # Training epochs
batch_size: 4                # Per-GPU batch size
grad_accum_steps: 4          # Gradient accumulation (effective batch = 16)
lr: 0.0001                   # Learning rate
early_stopping: true         # Stop if validation plateaus
early_stopping_patience: 10  # Epochs to wait
```

## EC2 Deployment

See [deploy/README.md](deploy/README.md) for full instructions.

```bash
# On a fresh EC2 GPU instance:
git clone https://github.com/sziegler11/rf-detr-playground.git
cd rf-detr-playground
chmod +x deploy/setup_ec2.sh
./deploy/setup_ec2.sh
```

Recommended: `g4dn.xlarge` ($0.53/hr) or `g5.xlarge` ($1.01/hr) with the AWS Deep Learning AMI.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test
```

## License

MIT
