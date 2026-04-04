"""Evaluation pipeline — run inference and compute COCO metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def load_model(checkpoint_path: str, model_size: str = "base"):
    """Load an RF-DETR model from a checkpoint."""
    from rf_detr_playground.training.trainer import get_model

    model = get_model(model_size)

    # RF-DETR loads weights via the checkpoint_path parameter
    # The exact API depends on rfdetr version
    try:
        import torch
        model.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    except Exception:
        # Fallback: some versions use a different loading mechanism
        print(f"Note: Loading checkpoint from {checkpoint_path}")
        model.checkpoint = checkpoint_path

    return model


def run_coco_eval(
    predictions_json: Path,
    ground_truth_json: Path,
) -> dict:
    """Run official COCO evaluation using pycocotools.

    Args:
        predictions_json: Path to predictions in COCO results format.
        ground_truth_json: Path to ground truth COCO annotations.

    Returns:
        Dict with mAP metrics.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(ground_truth_json))
    coco_dt = coco_gt.loadRes(str(predictions_json))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "mAP_50_95": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
        "AR_small": float(coco_eval.stats[9]),
        "AR_medium": float(coco_eval.stats[10]),
        "AR_large": float(coco_eval.stats[11]),
    }

    return results


def compute_per_class_ap(
    predictions_json: Path,
    ground_truth_json: Path,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute per-class AP@50 using pycocotools.

    Returns a dict mapping class name/id to AP value.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(ground_truth_json))
    coco_dt = coco_gt.loadRes(str(predictions_json))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [0.5]
    coco_eval.evaluate()
    coco_eval.accumulate()

    per_class = {}
    cat_ids = coco_gt.getCatIds()
    cat_info = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    for idx, cat_id in enumerate(coco_eval.params.catIds):
        # precision shape: [T, R, K, A, M] — we want T=0, A=0, M=-1
        precision = coco_eval.eval["precision"][0, :, idx, 0, -1]
        ap = float(precision[precision > -1].mean()) if (precision > -1).any() else 0.0

        name = cat_info.get(cat_id, str(cat_id))
        if class_names and idx < len(class_names):
            name = class_names[idx]
        per_class[name] = ap

    return per_class


def evaluate_model(
    checkpoint_path: str,
    dataset_dir: str,
    model_size: str = "base",
    output_dir: str = "outputs/eval",
    confidence_threshold: float = 0.3,
) -> dict:
    """Full evaluation pipeline: load model, run inference, compute metrics.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file).
        dataset_dir: Path to dataset directory with valid/ split.
        model_size: Model size (nano/small/medium/base/large).
        output_dir: Directory to save evaluation results.
        confidence_threshold: Min confidence for predictions.

    Returns:
        Dict containing all evaluation metrics.
    """
    import supervision as sv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find ground truth annotations
    valid_dir = Path(dataset_dir) / "valid"
    if not valid_dir.exists():
        valid_dir = Path(dataset_dir) / "test"
    gt_json = valid_dir / "_annotations.coco.json"

    if not gt_json.exists():
        raise FileNotFoundError(f"Ground truth annotations not found at {gt_json}")

    with open(gt_json) as f:
        gt_data = json.load(f)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    from rf_detr_playground.training.trainer import get_model
    model = get_model(model_size)

    # Try to load checkpoint weights
    try:
        import torch
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Could not load checkpoint ({e}). Using pretrained weights.")

    # Run inference on validation images
    print("Running inference on validation set...")
    predictions = []
    inference_times = []

    for img_info in gt_data["images"]:
        img_path = valid_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        start = time.time()
        detections = model.predict(str(img_path), threshold=confidence_threshold)
        elapsed = time.time() - start
        inference_times.append(elapsed)

        # Convert supervision Detections to COCO results format
        if detections is not None and len(detections) > 0:
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                score = float(detections.confidence[i])
                class_id = int(detections.class_id[i])

                predictions.append({
                    "image_id": img_info["id"],
                    "category_id": class_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": score,
                })

    # Save predictions
    pred_json = output_path / "predictions.json"
    with open(pred_json, "w") as f:
        json.dump(predictions, f)

    # Compute COCO metrics
    print("Computing COCO metrics...")
    results = {}

    if predictions:
        results["coco_metrics"] = run_coco_eval(pred_json, gt_json)

        # Per-class AP
        class_names = [c["name"] for c in gt_data.get("categories", [])]
        results["per_class_ap"] = compute_per_class_ap(pred_json, gt_json, class_names)
    else:
        print("Warning: No predictions generated. All metrics will be zero.")
        results["coco_metrics"] = {k: 0.0 for k in [
            "mAP_50_95", "mAP_50", "mAP_75", "mAP_small", "mAP_medium",
            "mAP_large", "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
        ]}
        results["per_class_ap"] = {}

    # Inference speed stats
    if inference_times:
        results["inference"] = {
            "avg_ms": float(np.mean(inference_times) * 1000),
            "median_ms": float(np.median(inference_times) * 1000),
            "total_images": len(inference_times),
        }

    # Save results
    results_json = output_path / "results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if "coco_metrics" in results:
        m = results["coco_metrics"]
        print(f"  mAP@50:95  = {m['mAP_50_95']:.4f}")
        print(f"  mAP@50     = {m['mAP_50']:.4f}")
        print(f"  mAP@75     = {m['mAP_75']:.4f}")
        print(f"  mAP-small  = {m['mAP_small']:.4f}")
        print(f"  mAP-medium = {m['mAP_medium']:.4f}")
        print(f"  mAP-large  = {m['mAP_large']:.4f}")
    if "per_class_ap" in results and results["per_class_ap"]:
        print("\nPer-class AP@50:")
        for cls, ap in sorted(results["per_class_ap"].items(), key=lambda x: -x[1]):
            print(f"  {cls:20s} {ap:.4f}")
    if "inference" in results:
        print(f"\nInference: {results['inference']['avg_ms']:.1f}ms avg ({results['inference']['total_images']} images)")
    print("=" * 60)
    print(f"\nResults saved to {results_json}")

    return results
