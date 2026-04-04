#!/usr/bin/env python3
"""Generate metric plots and detection visualizations from evaluation results."""

import click


@click.command()
@click.option("--results-dir", "-r", type=click.Path(exists=True), default="outputs",
              help="Directory containing evaluation results.json files")
@click.option("--output-dir", "-o", type=str, default=None,
              help="Output directory for plots (defaults to results-dir)")
def main(results_dir, output_dir):
    """Generate visualization plots from evaluation results."""
    import json
    from pathlib import Path

    from rf_detr_playground.visualization.plots import (
        plot_model_comparison,
        plot_per_class_ap,
    )

    results_path = Path(results_dir)
    out_path = Path(output_dir) if output_dir else results_path

    # Find all results.json files
    all_results = {}
    for result_file in sorted(results_path.rglob("results.json")):
        with open(result_file) as f:
            data = json.load(f)

        run_name = result_file.parent.name
        all_results[run_name] = data

        # Generate per-run plots
        run_out = out_path / run_name

        if "per_class_ap" in data and data["per_class_ap"]:
            plot_per_class_ap(
                data["per_class_ap"],
                run_out / "per_class_ap.png",
                title=f"Per-Class AP — {run_name}",
            )
            print(f"  Generated per-class AP chart for {run_name}")

    # Generate comparison if multiple runs exist
    coco_results = {}
    for name, data in all_results.items():
        if "coco_metrics" in data:
            coco_results[name] = data["coco_metrics"]

    if len(coco_results) > 1:
        plot_model_comparison(
            coco_results,
            out_path / "model_comparison.png",
        )
        print(f"\nGenerated model comparison chart ({len(coco_results)} runs)")

    if not all_results:
        print("No results.json files found in the results directory.")
        print("Run evaluation first: python scripts/evaluate.py ...")
    else:
        print(f"\nAll plots saved to {out_path}")


if __name__ == "__main__":
    main()
