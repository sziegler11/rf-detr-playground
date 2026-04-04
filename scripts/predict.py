#!/usr/bin/env python3
"""Run inference on images and visualize detections."""

import click


@click.command()
@click.option("--checkpoint", "-c", type=click.Path(), default=None,
              help="Path to model checkpoint (uses pretrained if not provided)")
@click.option("--model", "-m", type=click.Choice(["nano", "small", "medium", "base", "large"]),
              default="base", help="Model size")
@click.option("--images", "-i", type=click.Path(exists=True), required=True,
              help="Path to image file or directory of images")
@click.option("--output-dir", "-o", type=str, default="outputs/predictions",
              help="Output directory for visualized images")
@click.option("--confidence", type=float, default=0.5,
              help="Confidence threshold")
@click.option("--dataset", "-d", type=str, default=None,
              help="Dataset name for class labels")
def main(checkpoint, model, images, output_dir, confidence, dataset):
    """Run RF-DETR inference on images and save visualized results."""
    from pathlib import Path

    from rf_detr_playground.training.trainer import get_model
    from rf_detr_playground.visualization.detections import draw_detections

    # Get class names if dataset specified
    class_names = None
    if dataset:
        from rf_detr_playground.data.registry import get_dataset
        class_names = get_dataset(dataset).classes

    # Load model
    print(f"Loading RF-DETR {model}...")
    det_model = get_model(model)

    if checkpoint:
        try:
            import torch
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            det_model.model.load_state_dict(state_dict)
            print(f"Loaded checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Using pretrained weights.")

    # Collect images
    images_path = Path(images)
    if images_path.is_file():
        image_files = [images_path]
    else:
        image_files = sorted(
            f for f in images_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

    if not image_files:
        print("No images found!")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(image_files)} images...")

    for img_path in image_files:
        detections = det_model.predict(str(img_path), threshold=confidence)

        out_file = output_path / f"pred_{img_path.name}"
        draw_detections(
            image_path=img_path,
            detections=detections,
            class_names=class_names,
            output_path=out_file,
        )

        num_dets = len(detections) if detections is not None else 0
        print(f"  {img_path.name}: {num_dets} detections → {out_file}")

    print(f"\nAll predictions saved to {output_path}")


if __name__ == "__main__":
    main()
