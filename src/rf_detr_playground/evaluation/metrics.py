"""Metric computation helpers for object detection evaluation."""

from __future__ import annotations

import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    union_area = w1 * h1 + w2 * h2 - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Compute Average Precision using the 101-point interpolation (COCO style)."""
    recall_levels = np.linspace(0, 1, 101)
    interpolated = np.zeros_like(recall_levels)

    for i, r in enumerate(recall_levels):
        mask = recalls >= r
        if mask.any():
            interpolated[i] = precisions[mask].max()

    return interpolated.mean()


def compute_precision_recall(
    tp: np.ndarray,
    confidence: np.ndarray,
    num_gt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute precision and recall arrays from TP flags sorted by confidence.

    Args:
        tp: Boolean array of true positive flags (sorted by descending confidence).
        confidence: Confidence scores (sorted descending).
        num_gt: Total number of ground truth objects.

    Returns:
        Tuple of (precision, recall) arrays.
    """
    if num_gt == 0:
        return np.zeros(len(tp)), np.zeros(len(tp))

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(~tp)

    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / num_gt

    return precision, recall


def build_confusion_matrix(
    pred_classes: list[int],
    gt_classes: list[int],
    num_classes: int,
    pred_boxes: np.ndarray | None = None,
    gt_boxes: np.ndarray | None = None,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Build a confusion matrix for detection results.

    Rows = ground truth, Columns = predictions.
    Last row/col = background (missed detections / false positives).
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    if pred_boxes is not None and gt_boxes is not None and len(pred_boxes) > 0 and len(gt_boxes) > 0:
        matched_gt = set()

        for pi, (pbox, pcls) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou = 0
            best_gi = -1

            for gi, (gbox, gcls) in enumerate(zip(gt_boxes, gt_classes)):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi >= 0:
                matched_gt.add(best_gi)
                matrix[gt_classes[best_gi]][pcls] += 1
            else:
                # False positive
                matrix[num_classes][pcls] += 1

        # Missed ground truths
        for gi, gcls in enumerate(gt_classes):
            if gi not in matched_gt:
                matrix[gcls][num_classes] += 1
    else:
        # No boxes to match
        for gcls in gt_classes:
            matrix[gcls][num_classes] += 1
        for pcls in pred_classes:
            matrix[num_classes][pcls] += 1

    return matrix
