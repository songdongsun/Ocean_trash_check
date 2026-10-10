# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train a 22-class TrashCan-Instance YOLO segmentation baseline on a CUDA server.

Run from the repository root:
    python ocean_trash/src/model/train_baseline.py
    python ocean_trash/src/model/train_baseline.py --device 2 --batch 16

Requires CUDA-enabled PyTorch. The default device is GPU 0; no CPU fallback is used.
The first run downloads COCO-pretrained weights if unavailable locally.
Metrics, resolved training arguments, and checkpoints are saved under ocean_trash/runs/segment.
These are project baseline settings, not a reproduction of the author's Mask R-CNN experiment.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from ultralytics import YOLO  # noqa: E402


def parse_args():
    """Parse model, data, and training settings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo11n-seg.pt", help="COCO-pretrained segmentation checkpoint")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/trashcan_instance_yolo/trashcan_instance.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--patience", type=int, default=0, help="Stop after this many epochs without improvement; 0 disables early stopping"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="Training GPU index, e.g. 0 or 2,5")
    parser.add_argument("--workers", type=int, default=4, help="Data-loader workers per GPU")
    parser.add_argument("--name", default="trashcan_instance_yolo11n_baseline")
    return parser.parse_args()


def main():
    """Fine-tune a pretrained segmentation model on the official train/val split."""
    args = parse_args()
    model = YOLO(args.model)
    if model.task != "segment":
        raise ValueError("This baseline requires an instance-segmentation model, such as yolo11n-seg.pt.")
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.001,
        momentum=0.9,
        warmup_bias_lr=0.0,
        amp=True,
        seed=0,
        deterministic=True,
        val=True,
        plots=True,
        project=str(PROJECT_ROOT / "runs/segment"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
