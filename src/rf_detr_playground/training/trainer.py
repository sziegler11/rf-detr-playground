"""Training wrapper for RF-DETR models."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    """Training configuration loaded from YAML."""

    # Dataset
    dataset_name: str = "aquarium"
    dataset_dir: str | None = None  # Override auto-resolved path

    # Model
    model_size: str = "base"  # nano, small, medium, base, large

    # Training
    epochs: int = 50
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 0

    # Resolution
    img_size: int = 560

    # Checkpointing
    output_dir: str = "outputs"
    run_name: str | None = None

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.005

    # Logging
    tensorboard: bool = False
    wandb: bool = False
    wandb_project: str = "rf-detr-playground"

    # Extra kwargs passed to model.train()
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        known = {k: v for k, v in data.items() if k in known_fields}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        known["extra"] = extra

        return cls(**known)


def get_model(size: str):
    """Instantiate an RF-DETR model by size name."""
    import rfdetr

    models = {
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
        "base": rfdetr.RFDETRBase,
        "large": rfdetr.RFDETRLarge,
    }

    if size not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Unknown model size '{size}'. Available: {available}")

    return models[size]()


def resolve_dataset_dir(config: TrainingConfig) -> str:
    """Resolve the dataset directory from config."""
    if config.dataset_dir:
        return config.dataset_dir
    return str(Path("data") / config.dataset_name)


def train(config: TrainingConfig) -> Path:
    """Run RF-DETR training with the given config.

    Returns the output directory path.
    """
    # Resolve paths
    dataset_dir = resolve_dataset_dir(config)
    run_name = config.run_name or f"{config.model_size}_{config.dataset_name}_{int(time.time())}"
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(vars(config), f, default_flow_style=False)

    print(f"Training RF-DETR {config.model_size}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size} (accum: {config.grad_accum_steps})")
    print(f"  Learning rate: {config.lr}")
    print(f"  Output: {output_dir}")

    # Create model
    model = get_model(config.model_size)

    # Build training kwargs
    train_kwargs = {
        "dataset_dir": dataset_dir,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "lr": config.lr,
        "output_dir": str(output_dir),
    }

    if config.early_stopping:
        train_kwargs["early_stopping"] = True
        train_kwargs["early_stopping_patience"] = config.early_stopping_patience
        train_kwargs["early_stopping_min_delta"] = config.early_stopping_min_delta

    if config.tensorboard:
        train_kwargs["tensorboard"] = True

    if config.wandb:
        train_kwargs["wandb"] = True
        train_kwargs["project"] = config.wandb_project
        train_kwargs["run"] = run_name

    # Merge extra kwargs
    train_kwargs.update(config.extra)

    # Train
    model.train(**train_kwargs)

    # Save training metadata
    metadata = {
        "model_size": config.model_size,
        "dataset": config.dataset_name,
        "epochs": config.epochs,
        "run_name": run_name,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create/update latest symlink
    latest_link = Path(config.output_dir) / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_name)

    print(f"\nTraining complete. Output: {output_dir}")
    return output_dir
