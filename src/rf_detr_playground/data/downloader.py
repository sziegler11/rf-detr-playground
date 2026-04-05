"""Dataset download, extraction, and preparation logic."""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from .converter import voc_to_coco, yolo_to_coco
from .registry import DATASETS, DatasetInfo, get_dataset


def _download(url: str) -> io.BytesIO:
    """Download a URL into an in-memory buffer with a progress bar."""
    print(f"  Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc="  Downloading") as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            buf.write(chunk)
            pbar.update(len(chunk))
    buf.seek(0)
    return buf


def _download_and_extract(url: str, dest: Path) -> None:
    """Download an archive from *url* and extract it into *dest*.

    Supports .zip, .tar, .tar.gz, and .tgz archives.
    """
    dest.mkdir(parents=True, exist_ok=True)
    buf = _download(url)

    print("  Extracting ...")
    if url.endswith(".zip"):
        with zipfile.ZipFile(buf) as zf:
            zf.extractall(dest)
    elif url.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if url.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(fileobj=buf, mode=mode) as tf:
            tf.extractall(dest, filter="data")
    else:
        # Try zip first, fall back to tar
        try:
            with zipfile.ZipFile(buf) as zf:
                zf.extractall(dest)
        except zipfile.BadZipFile:
            buf.seek(0)
            with tarfile.open(fileobj=buf) as tf:
                tf.extractall(dest, filter="data")
    print("  Done.")


def _prepare_coco128(dataset_dir: Path, info: DatasetInfo) -> None:
    """Download COCO128 (YOLO format) and convert to COCO JSON."""
    raw_dir = dataset_dir / "_raw"
    _download_and_extract(info.url, raw_dir)

    # Ultralytics COCO128 extracts to coco128/images/train2017/ and coco128/labels/train2017/
    extracted = raw_dir / "coco128"
    img_src = extracted / "images" / "train2017"
    lbl_src = extracted / "labels" / "train2017"

    # Set up COCO-style directory structure
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    img_dest = train_dir / "images"
    if img_src.exists():
        shutil.copytree(img_src, img_dest, dirs_exist_ok=True)

    # Convert YOLO labels to COCO JSON
    if lbl_src.exists() and img_dest.exists():
        yolo_to_coco(img_dest, lbl_src, info.classes, train_dir / "annotations.json")

    # Clean up raw download
    shutil.rmtree(raw_dir, ignore_errors=True)


def _prepare_pascal_voc(dataset_dir: Path, info: DatasetInfo) -> None:
    """Download Pascal VOC and convert XML annotations to COCO JSON."""
    raw_dir = dataset_dir / "_raw"
    _download_and_extract(info.url, raw_dir)

    # Find the VOCdevkit root (may be nested)
    voc_root = None
    for candidate in raw_dir.rglob("VOCdevkit"):
        voc_root = candidate
        break

    if voc_root is None:
        # Fallback: look for Annotations directory directly
        for candidate in raw_dir.rglob("Annotations"):
            voc_root = candidate.parent
            break

    if voc_root is None:
        raise FileNotFoundError("Could not find VOCdevkit or Annotations in download")

    # Find the year directory (e.g., VOC2012)
    voc_year = None
    for d in voc_root.iterdir():
        if d.is_dir() and d.name.startswith("VOC"):
            voc_year = d
            break
    if voc_year is None:
        voc_year = voc_root

    img_src = voc_year / "JPEGImages"
    ann_src = voc_year / "Annotations"
    imagesets = voc_year / "ImageSets" / "Main"

    for split in info.splits:
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Read image list for this split if available
        split_file = imagesets / f"{split}.txt" if imagesets.exists() else None
        image_ids: set[str] | None = None
        if split_file and split_file.exists():
            image_ids = {
                line.strip().split()[0]
                for line in split_file.read_text().splitlines()
                if line.strip()
            }

        # Copy images for this split
        split_img_dir = split_dir / "images"
        split_ann_dir = split_dir / "annotations_voc"
        split_img_dir.mkdir(exist_ok=True)
        split_ann_dir.mkdir(exist_ok=True)

        for xml_path in sorted(ann_src.glob("*.xml")):
            stem = xml_path.stem
            if image_ids is not None and stem not in image_ids:
                continue

            # Copy annotation
            shutil.copy2(xml_path, split_ann_dir / xml_path.name)

            # Copy corresponding image
            for ext in (".jpg", ".jpeg", ".png"):
                img_path = img_src / (stem + ext)
                if img_path.exists():
                    shutil.copy2(img_path, split_img_dir / img_path.name)
                    break

        # Convert to COCO JSON
        voc_to_coco(
            split_img_dir, split_ann_dir, info.classes,
            split_dir / "annotations.json",
        )

    shutil.rmtree(raw_dir, ignore_errors=True)


def _prepare_aquarium(dataset_dir: Path, info: DatasetInfo) -> None:
    """Download Aquarium dataset via the Roboflow SDK.

    Requires the ``roboflow`` package and a Roboflow API key set via the
    ``ROBOFLOW_API_KEY`` environment variable (free at https://roboflow.com).
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise RuntimeError(
            "The 'roboflow' package is required to download the Aquarium dataset.\n"
            "Install it with:  pip install roboflow\n"
            "Then set your API key:  export ROBOFLOW_API_KEY=<your-key>\n"
            "Get a free key at https://app.roboflow.com/settings/api"
        )

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Set the ROBOFLOW_API_KEY environment variable to download the Aquarium dataset.\n"
            "Get a free key at https://app.roboflow.com/settings/api\n"
            "  export ROBOFLOW_API_KEY=<your-key>"
        )

    print("  Downloading Aquarium dataset via Roboflow SDK ...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("brad-dwyer").project("aquarium-combined")
    ds = project.version(2).download("coco", location=str(dataset_dir), overwrite=True)

    # Normalize annotation filenames to annotations.json
    for split in info.splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        rf_ann = split_dir / "_annotations.coco.json"
        target = split_dir / "annotations.json"
        if rf_ann.exists() and not target.exists():
            rf_ann.rename(target)


# Map dataset names to their preparation functions
_PREPARERS = {
    "coco128": _prepare_coco128,
    "pascal_voc": _prepare_pascal_voc,
    "aquarium": _prepare_aquarium,
}


def download_dataset(name: str, data_root: Path) -> Path:
    """Download and prepare a single dataset.

    Args:
        name: Registered dataset name.
        data_root: Root directory for all datasets.

    Returns:
        Path to the prepared dataset directory.
    """
    info = get_dataset(name)
    dataset_dir = data_root / name

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset '{name}' already exists at {dataset_dir}, skipping download.")
        return dataset_dir

    print(f"Preparing dataset: {name}")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    preparer = _PREPARERS.get(name)
    if preparer is None:
        raise ValueError(
            f"No download handler for '{name}'. "
            f"Available: {list(_PREPARERS.keys())}"
        )

    preparer(dataset_dir, info)
    print(f"Dataset '{name}' ready at {dataset_dir}")
    return dataset_dir


def download_all(data_root: Path) -> list[Path]:
    """Download and prepare all registered datasets."""
    paths = []
    for name in DATASETS:
        try:
            paths.append(download_dataset(name, data_root))
        except Exception as e:
            print(f"Warning: failed to download '{name}': {e}")
    return paths
