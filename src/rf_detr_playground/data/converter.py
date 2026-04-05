"""Format converters: VOC XML and YOLO txt to COCO JSON."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def voc_to_coco(
    img_dir: Path,
    ann_dir: Path,
    classes: list[str],
    output_json: Path,
) -> dict:
    """Convert Pascal VOC XML annotations to COCO JSON format.

    Args:
        img_dir: Directory containing images.
        ann_dir: Directory containing VOC XML annotation files.
        classes: Ordered list of class names (index = category_id).
        output_json: Path to write the output COCO JSON file.

    Returns:
        The COCO annotation dict.
    """
    class_to_id = {name: i for i, name in enumerate(classes)}

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name} for i, name in enumerate(classes)
        ],
    }

    ann_id = 0
    for img_id, xml_path in enumerate(sorted(ann_dir.glob("*.xml"))):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename", "")
        size = root.find("size")
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))

        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        for obj in root.findall("object"):
            name = obj.findtext("name", "")
            if name not in class_to_id:
                continue

            bbox_el = obj.find("bndbox")
            xmin = float(bbox_el.findtext("xmin", "0"))
            ymin = float(bbox_el.findtext("ymin", "0"))
            xmax = float(bbox_el.findtext("xmax", "0"))
            ymax = float(bbox_el.findtext("ymax", "0"))

            w = xmax - xmin
            h = ymax - ymin

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_to_id[name],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    return coco


def yolo_to_coco(
    img_dir: Path,
    lbl_dir: Path,
    classes: list[str],
    output_json: Path,
) -> dict:
    """Convert YOLO-format labels to COCO JSON format.

    YOLO labels are text files with one line per object:
        class_id cx cy w h  (all normalized 0-1)

    Args:
        img_dir: Directory containing images.
        lbl_dir: Directory containing YOLO .txt label files.
        classes: Ordered list of class names (index = category_id).
        output_json: Path to write the output COCO JSON file.

    Returns:
        The COCO annotation dict.
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name} for i, name in enumerate(classes)
        ],
    }

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in img_extensions
    )

    ann_id = 0
    for img_id, img_path in enumerate(image_files):
        img = Image.open(img_path)
        img_w, img_h = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            # Convert normalized center coords to pixel top-left coords
            x = (cx - bw / 2) * img_w
            y = (cy - bh / 2) * img_h
            w = bw * img_w
            h = bh * img_h

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    return coco
