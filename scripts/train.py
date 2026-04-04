#!/usr/bin/env python3
"""Train an RF-DETR model on a dataset."""

import click


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True,
              help="Path to YAML config file")
@click.option("--model", "-m", type=click.Choice(["nano", "small", "medium", "base", "large"]),
              default=None, help="Override model size from config")
@click.option("--epochs", "-e", type=int, default=None, help="Override epochs from config")
@click.option("--batch-size", "-b", type=int, default=None, help="Override batch size")
@click.option("--lr", type=float, default=None, help="Override learning rate")
@click.option("--run-name", type=str, default=None, help="Name for this training run")
@click.option("--output-dir", type=str, default=None, help="Override output directory")
def main(config, model, epochs, batch_size, lr, run_name, output_dir):
    """Train an RF-DETR model using a YAML configuration file."""
    from rf_detr_playground.training.trainer import TrainingConfig, train

    cfg = TrainingConfig.from_yaml(config)

    # Apply CLI overrides
    if model:
        cfg.model_size = model
    if epochs:
        cfg.epochs = epochs
    if batch_size:
        cfg.batch_size = batch_size
    if lr:
        cfg.lr = lr
    if run_name:
        cfg.run_name = run_name
    if output_dir:
        cfg.output_dir = output_dir

    train(cfg)


if __name__ == "__main__":
    main()
