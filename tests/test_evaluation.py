"""Tests for the evaluation and metrics modules."""

import numpy as np
import pytest

from rf_detr_playground.evaluation.metrics import (
    build_confusion_matrix,
    compute_ap,
    compute_iou,
    compute_precision_recall,
)


class TestIoU:
    def test_perfect_overlap(self):
        box = np.array([0, 0, 100, 100])
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 50, 50])
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 100, 100])
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        assert compute_iou(box1, box2) == pytest.approx(2500 / 17500)

    def test_contained_box(self):
        outer = np.array([0, 0, 100, 100])
        inner = np.array([25, 25, 50, 50])
        # Intersection: 50x50 = 2500
        # Union: 10000 + 2500 - 2500 = 10000
        assert compute_iou(outer, inner) == pytest.approx(2500 / 10000)

    def test_zero_area_box(self):
        box1 = np.array([0, 0, 0, 0])
        box2 = np.array([0, 0, 10, 10])
        assert compute_iou(box1, box2) == pytest.approx(0.0)


class TestPrecisionRecall:
    def test_all_true_positives(self):
        tp = np.array([True, True, True])
        conf = np.array([0.9, 0.8, 0.7])
        prec, rec = compute_precision_recall(tp, conf, num_gt=3)
        assert prec[-1] == pytest.approx(1.0)
        assert rec[-1] == pytest.approx(1.0)

    def test_all_false_positives(self):
        tp = np.array([False, False, False])
        conf = np.array([0.9, 0.8, 0.7])
        prec, rec = compute_precision_recall(tp, conf, num_gt=3)
        assert prec[-1] == pytest.approx(0.0)
        assert rec[-1] == pytest.approx(0.0)

    def test_mixed(self):
        tp = np.array([True, False, True])
        conf = np.array([0.9, 0.8, 0.7])
        prec, rec = compute_precision_recall(tp, conf, num_gt=3)

        # After 1st: prec=1/1, rec=1/3
        assert prec[0] == pytest.approx(1.0)
        assert rec[0] == pytest.approx(1 / 3)

        # After 2nd: prec=1/2, rec=1/3
        assert prec[1] == pytest.approx(0.5)

        # After 3rd: prec=2/3, rec=2/3
        assert prec[2] == pytest.approx(2 / 3)
        assert rec[2] == pytest.approx(2 / 3)

    def test_no_ground_truth(self):
        tp = np.array([False])
        conf = np.array([0.9])
        prec, rec = compute_precision_recall(tp, conf, num_gt=0)
        assert prec[0] == pytest.approx(0.0)
        assert rec[0] == pytest.approx(0.0)


class TestAP:
    def test_perfect_ap(self):
        prec = np.ones(10)
        rec = np.linspace(0, 1, 10)
        ap = compute_ap(prec, rec)
        assert ap == pytest.approx(1.0)

    def test_zero_ap(self):
        prec = np.zeros(10)
        rec = np.linspace(0, 1, 10)
        ap = compute_ap(prec, rec)
        assert ap == pytest.approx(0.0)


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        pred_classes = [0, 1, 0]
        gt_classes = [0, 1, 0]
        pred_boxes = np.array([[0, 0, 100, 100], [200, 200, 100, 100], [400, 0, 50, 50]])
        gt_boxes = np.array([[0, 0, 100, 100], [200, 200, 100, 100], [400, 0, 50, 50]])

        matrix = build_confusion_matrix(pred_classes, gt_classes, 2, pred_boxes, gt_boxes)
        assert matrix[0, 0] == 2  # Two correct class 0
        assert matrix[1, 1] == 1  # One correct class 1
        assert matrix[2, :].sum() == 0  # No false positives from background

    def test_empty_predictions(self):
        matrix = build_confusion_matrix([], [0, 1], 2)
        assert matrix[0, 2] == 1  # class 0 missed
        assert matrix[1, 2] == 1  # class 1 missed
