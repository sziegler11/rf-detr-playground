"""Dataset registry mapping names to metadata and download info."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetInfo:
    """Metadata for a registered dataset."""

    name: str
    num_classes: int
    classes: list[str]
    format: str  # "coco", "yolo", or "voc"
    splits: list[str]
    url: str | None = None
    description: str = ""


# --- COCO 128-image sample (YOLO format from Ultralytics mirror) ---

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# --- Pascal VOC 2012 ---

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# --- Aquarium (Roboflow Universe) ---

AQUARIUM_CLASSES = [
    "fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray",
]

# ---------------------------------------------------------------------------

DATASETS: dict[str, DatasetInfo] = {
    "coco128": DatasetInfo(
        name="coco128",
        num_classes=80,
        classes=COCO_CLASSES,
        format="yolo",
        splits=["train"],
        url="https://ultralytics.com/assets/coco128.zip",
        description="128-image COCO sample for sanity checks (YOLO format)",
    ),
    "pascal_voc": DatasetInfo(
        name="pascal_voc",
        num_classes=20,
        classes=VOC_CLASSES,
        format="voc",
        splits=["train", "val"],
        url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        description="Pascal VOC 2012 — 20-class detection benchmark",
    ),
    "aquarium": DatasetInfo(
        name="aquarium",
        num_classes=7,
        classes=AQUARIUM_CLASSES,
        format="coco",
        splits=["train", "valid", "test"],
        url=None,  # Downloaded via Roboflow SDK (requires ROBOFLOW_API_KEY)
        description="Aquarium dataset — 7-class underwater detection from Roboflow",
    ),
}


def list_datasets() -> list[str]:
    """Return all registered dataset names."""
    return list(DATASETS.keys())


def get_dataset(name: str) -> DatasetInfo:
    """Look up a dataset by name. Raises KeyError if not found."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: '{name}'. Available: {list_datasets()}")
    return DATASETS[name]
