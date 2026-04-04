#!/usr/bin/env python3
"""Benchmark and compare multiple RF-DETR model runs."""

import click


@click.command()
@click.option("--results-dir", "-r", type=click.Path(exists=True), default="outputs",
              help="Directory containing evaluation results")
@click.option("--output-dir", "-o", type=str, default="outputs/benchmark",
              help="Output directory for comparison plots")
@click.option("--checkpoints", "-c", type=str, multiple=True,
              help="Specific checkpoint paths to evaluate (repeatable)")
@click.option("--dataset", "-d", type=str, default="aquarium",
              help="Dataset to evaluate on")
@click.option("--models", "-m", type=str, multiple=True,
              help="Model sizes to compare (repeatable, e.g. -m base -m large)")
def main(results_dir, output_dir, checkpoints, dataset, models):
    """Compare multiple RF-DETR models or training runs."""
    import json
    from pathlib import Path

    from rf_detr_playground.visualization.plots import plot_model_comparison, plot_per_class_ap

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    if checkpoints:
        # Evaluate specific checkpoints
        from rf_detr_playground.evaluation.evaluator import evaluate_model

        for cp in checkpoints:
            cp_path = Path(cp)
            run_name = cp_path.stem

            # Try to infer model size from metadata
            model_size = "base"
            metadata_path = cp_path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    meta = json.load(f)
                model_size = meta.get("model_size", "base")

            dataset_dir = f"data/{dataset}"
            eval_results = evaluate_model(
                checkpoint_path=str(cp),
                dataset_dir=dataset_dir,
                model_size=model_size,
                output_dir=str(output_path / run_name),
            )
            results[run_name] = eval_results.get("coco_metrics", {})

    elif models:
        # Compare pretrained model sizes
        from rf_detr_playground.evaluation.evaluator import evaluate_model

        for model_size in models:
            print(f"\nEvaluating RF-DETR {model_size} (pretrained)...")
            dataset_dir = f"data/{dataset}"
            eval_results = evaluate_model(
                checkpoint_path="",  # Use pretrained
                dataset_dir=dataset_dir,
                model_size=model_size,
                output_dir=str(output_path / model_size),
            )
            results[f"RF-DETR {model_size}"] = eval_results.get("coco_metrics", {})

    else:
        # Scan results directory for existing evaluations
        results_path = Path(results_dir)
        for result_file in sorted(results_path.rglob("results.json")):
            with open(result_file) as f:
                data = json.load(f)

            run_name = result_file.parent.name
            if "coco_metrics" in data:
                results[run_name] = data["coco_metrics"]

    if not results:
        print("No results found to compare.")
        print("\nTo benchmark, either:")
        print("  1. Run evaluations first: python scripts/evaluate.py ...")
        print("  2. Specify checkpoints: python scripts/benchmark.py -c path/to/best.pt")
        print("  3. Compare model sizes: python scripts/benchmark.py -m base -m large")
        return

    # Generate comparison plot
    plot_model_comparison(results, output_path / "model_comparison.png")
    print(f"\nComparison chart saved to {output_path / 'model_comparison.png'}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Run':<25} {'mAP@50:95':>10} {'mAP@50':>10} {'mAP@75':>10}")
    print("-" * 70)
    for name, metrics in results.items():
        print(
            f"{name:<25} "
            f"{metrics.get('mAP_50_95', 0):>10.4f} "
            f"{metrics.get('mAP_50', 0):>10.4f} "
            f"{metrics.get('mAP_75', 0):>10.4f}"
        )
    print("=" * 70)

    # Save comparison data
    with open(output_path / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison data saved to {output_path / 'comparison.json'}")


if __name__ == "__main__":
    main()
