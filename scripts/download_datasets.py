#!/usr/bin/env python3
"""Download and prepare datasets for RF-DETR training."""

import click

from rf_detr_playground.data.downloader import download_all, download_dataset
from rf_detr_playground.data.registry import list_datasets


@click.command()
@click.option("--dataset", "-d", type=str, help="Dataset name to download")
@click.option("--all", "download_all_flag", is_flag=True, help="Download all datasets")
@click.option("--data-root", type=click.Path(), default="data", help="Root directory for datasets")
def main(dataset: str | None, download_all_flag: bool, data_root: str):
    """Download and prepare datasets for RF-DETR training."""
    from pathlib import Path

    root = Path(data_root)

    if download_all_flag:
        print("Downloading all datasets...")
        download_all(root)
    elif dataset:
        download_dataset(dataset, root)
    else:
        print("Available datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        print("\nUsage:")
        print("  python scripts/download_datasets.py --dataset aquarium")
        print("  python scripts/download_datasets.py --all")


if __name__ == "__main__":
    main()
