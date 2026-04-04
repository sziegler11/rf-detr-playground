"""Tests for the dataset pipeline."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from rf_detr_playground.data.registry import DATASETS, get_dataset, list_datasets
from rf_detr_playground.data.converter import voc_to_coco, yolo_to_coco


class TestRegistry:
    def test_list_datasets_returns_all(self):
        names = list_datasets()
        assert "coco128" in names
        assert "pascal_voc" in names
        assert "aquarium" in names

    def test_get_dataset_returns_info(self):
        info = get_dataset("aquarium")
        assert info.name == "aquarium"
        assert info.num_classes == 7
        assert len(info.classes) == 7
        assert "fish" in info.classes

    def test_get_dataset_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            get_dataset("nonexistent")

    def test_all_datasets_have_required_fields(self):
        for name, info in DATASETS.items():
            assert info.name == name
            assert info.num_classes > 0
            assert len(info.classes) == info.num_classes
            assert info.format in ("coco", "yolo", "voc")
            assert len(info.splits) > 0


class TestVocToCoco:
    def test_converts_voc_xml(self, tmp_path):
        # Create a minimal VOC XML annotation
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()

        # Create a dummy image file
        (img_dir / "test.jpg").write_bytes(b"fake image")

        # Create a VOC XML annotation
        xml_content = """<?xml version="1.0"?>
<annotation>
    <filename>test.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>cat</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>300</xmax>
            <ymax>300</ymax>
        </bndbox>
    </object>
    <object>
        <name>dog</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>200</xmin>
            <ymin>200</ymin>
            <xmax>400</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>"""
        (ann_dir / "test.xml").write_text(xml_content)

        output_json = tmp_path / "output.json"
        classes = ["cat", "dog"]
        result = voc_to_coco(img_dir, ann_dir, classes, output_json)

        assert output_json.exists()
        assert len(result["images"]) == 1
        assert len(result["annotations"]) == 2
        assert len(result["categories"]) == 2

        # Check bbox format (x, y, w, h)
        ann = result["annotations"][0]
        assert ann["bbox"] == [100, 100, 200, 200]
        assert ann["category_id"] == 0  # cat

    def test_skips_unknown_classes(self, tmp_path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()

        (img_dir / "test.jpg").write_bytes(b"fake")
        xml_content = """<?xml version="1.0"?>
<annotation>
    <filename>test.jpg</filename>
    <size><width>100</width><height>100</height></size>
    <object>
        <name>unknown_class</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
    </object>
</annotation>"""
        (ann_dir / "test.xml").write_text(xml_content)

        output_json = tmp_path / "output.json"
        result = voc_to_coco(img_dir, ann_dir, ["cat"], output_json)
        assert len(result["annotations"]) == 0


class TestYoloToCoco:
    def test_converts_yolo_labels(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        # Create a small test image
        from PIL import Image
        img = Image.new("RGB", (640, 480))
        img.save(img_dir / "test.jpg")

        # YOLO label: class cx cy w h (normalized)
        (lbl_dir / "test.txt").write_text("0 0.5 0.5 0.3125 0.4167\n1 0.25 0.25 0.15625 0.2083\n")

        output_json = tmp_path / "output.json"
        classes = ["cat", "dog"]
        result = yolo_to_coco(img_dir, lbl_dir, classes, output_json)

        assert output_json.exists()
        assert len(result["images"]) == 1
        assert len(result["annotations"]) == 2

        # Check conversion: cx=0.5, cy=0.5, w=0.3125, h=0.4167 on 640x480
        ann = result["annotations"][0]
        assert ann["category_id"] == 0
        assert abs(ann["bbox"][2] - 200) < 1  # w = 0.3125 * 640 = 200
        assert abs(ann["bbox"][3] - 200) < 1  # h = 0.4167 * 480 ≈ 200
