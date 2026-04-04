#!/usr/bin/env python3
"""Evaluate a trained RF-DETR model."""

import click


@click.command()
@click.option("--checkpoint", "-c", type=click.Path(exists=True), required=True,
              help="Path to model checkpoint (.pt file)")
@click.option("--dataset", "-d", type=str, required=True,
              help="Dataset name or path to dataset directory")
@click.option("--model", "-m", type=click.Choice(["nano", "small", "medium", "base", "large"]),
              default="base", help="Model size")
@click.option("--output-dir", "-o", type=str, default=None,
              help="Output directory for results")
@click.option("--confidence", type=float, default=0.3,
              help="Confidence threshold for predictions")
@click.option("--visualize/--no-visualize", default=True,
              help="Generate visualization plots")
def main(checkpoint, dataset, model, output_dir, confidence, visualize):
    """Evaluate a trained RF-DETR model on a dataset."""
    import json
    from pathlib import Path

    from rf_detr_playground.evaluation.evaluator import evaluate_model

    # Resolve dataset dir
    dataset_dir = dataset
    if not Path(dataset).exists():
        dataset_dir = f"data/{dataset}"

    if output_dir is None:
        output_dir = f"outputs/eval_{Path(checkpoint).stem}"

    results = evaluate_model(
        checkpoint_path=checkpoint,
        dataset_dir=dataset_dir,
        model_size=model,
        output_dir=output_dir,
        confidence_threshold=confidence,
    )

    if visualize and results.get("per_class_ap"):
        from rf_detr_playground.visualization.plots import (
            plot_per_class_ap,
            plot_model_comparison,
        )

        output_path = Path(output_dir)

        # Per-class AP chart
        plot_per_class_ap(
            results["per_class_ap"],
            output_path / "per_class_ap.png",
        )
        print(f"Saved per-class AP chart to {output_path / 'per_class_ap.png'}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
