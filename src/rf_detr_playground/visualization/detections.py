"""Detection visualization — draw bounding boxes on images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def draw_detections(
    image_path: str | Path,
    detections,
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Draw bounding box detections on an image using supervision.

    Args:
        image_path: Path to the source image.
        detections: supervision.Detections object.
        class_names: List of class names for label display.
        output_path: If provided, save the annotated image here.

    Returns:
        Annotated image as numpy array.
    """
    import supervision as sv

    image = np.array(Image.open(image_path).convert("RGB"))

    # Create annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Build labels
    labels = []
    if detections is not None and len(detections) > 0:
        for i in range(len(detections)):
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0
            name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
            labels.append(f"{name} {conf:.2f}")

        annotated = box_annotator.annotate(image.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)
    else:
        annotated = image.copy()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(annotated).save(output_path)

    return annotated


def draw_gt_vs_pred(
    image_path: str | Path,
    gt_detections,
    pred_detections,
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Create a side-by-side comparison of ground truth vs predictions.

    Args:
        image_path: Path to the source image.
        gt_detections: Ground truth supervision.Detections.
        pred_detections: Predicted supervision.Detections.
        class_names: List of class names.
        output_path: If provided, save the comparison image here.

    Returns:
        Side-by-side annotated image as numpy array.
    """
    import supervision as sv

    image = np.array(Image.open(image_path).convert("RGB"))

    # Annotate ground truth
    gt_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
    gt_label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, color=sv.Color.GREEN)

    gt_labels = []
    if gt_detections is not None and len(gt_detections) > 0:
        for i in range(len(gt_detections)):
            cls_id = int(gt_detections.class_id[i]) if gt_detections.class_id is not None else 0
            name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
            gt_labels.append(name)
        gt_image = gt_annotator.annotate(image.copy(), gt_detections)
        gt_image = gt_label_annotator.annotate(gt_image, gt_detections, labels=gt_labels)
    else:
        gt_image = image.copy()

    # Annotate predictions
    pred_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.RED)
    pred_label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, color=sv.Color.RED)

    pred_labels = []
    if pred_detections is not None and len(pred_detections) > 0:
        for i in range(len(pred_detections)):
            cls_id = int(pred_detections.class_id[i]) if pred_detections.class_id is not None else 0
            conf = float(pred_detections.confidence[i]) if pred_detections.confidence is not None else 0
            name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
            pred_labels.append(f"{name} {conf:.2f}")
        pred_image = pred_annotator.annotate(image.copy(), pred_detections)
        pred_image = pred_label_annotator.annotate(pred_image, pred_detections, labels=pred_labels)
    else:
        pred_image = image.copy()

    # Create side-by-side
    h, w = image.shape[:2]
    combined = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    combined[:, :w] = gt_image
    combined[:, w + 20:] = pred_image

    # Add titles using PIL
    from PIL import ImageDraw, ImageFont

    pil_combined = Image.fromarray(combined)
    draw = ImageDraw.Draw(pil_combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((10, 5), "Ground Truth", fill=(0, 255, 0), font=font)
    draw.text((w + 30, 5), "Predictions", fill=(255, 0, 0), font=font)

    result = np.array(pil_combined)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_combined.save(output_path)

    return result


def visualize_dataset_sample(
    dataset_dir: str | Path,
    num_images: int = 9,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Visualize a grid of sample images with GT annotations from a dataset.

    Args:
        dataset_dir: Path to dataset directory with train/ or valid/ split.
        num_images: Number of images to show in the grid.
        output_path: If provided, save the grid image here.

    Returns:
        Grid image as numpy array.
    """
    import json
    import math

    import supervision as sv

    dataset_path = Path(dataset_dir)
    split_dir = dataset_path / "train"
    if not split_dir.exists():
        split_dir = dataset_path / "valid"

    ann_path = split_dir / "_annotations.coco.json"
    with open(ann_path) as f:
        coco = json.load(f)

    class_names = [c["name"] for c in coco["categories"]]

    # Select sample images
    images = coco["images"][:num_images]
    cols = min(3, len(images))
    rows = math.ceil(len(images) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, img_info in enumerate(images):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]

        img_path = split_dir / img_info["file_name"]
        if not img_path.exists():
            ax.set_visible(False)
            continue

        img = np.array(Image.open(img_path).convert("RGB"))

        # Get annotations for this image
        img_anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]

        if img_anns:
            xyxy = []
            class_ids = []
            for ann in img_anns:
                x, y, w, h = ann["bbox"]
                xyxy.append([x, y, x + w, y + h])
                class_ids.append(ann["category_id"])

            detections = sv.Detections(
                xyxy=np.array(xyxy),
                class_id=np.array(class_ids),
            )

            box_annotator = sv.BoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)
            labels = [class_names[cid] if cid < len(class_names) else str(cid) for cid in class_ids]

            img = box_annotator.annotate(img, detections)
            img = label_annotator.annotate(img, detections, labels=labels)

        ax.imshow(img)
        ax.set_title(img_info["file_name"], fontsize=8)
        ax.axis("off")

    # Hide empty subplots
    for idx in range(len(images), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].set_visible(False)

    plt.suptitle(f"Dataset Sample ({dataset_path.name})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    result = fig_to_array(fig)
    plt.close(fig)
    return result


def fig_to_array(fig) -> np.ndarray:
    """Convert a matplotlib figure to a numpy array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.asarray(buf)[:, :, :3]


# Import matplotlib here to avoid import at module level for non-viz code
import matplotlib.pyplot as plt
