# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics.data.converter import convert_coco  # noqa: E402


def parse_args():
    """Parse dataset conversion paths."""
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Convert TrashCan COCO instance annotations to YOLO-Seg format.")
    parser.add_argument("--raw-dir", type=Path, default=project_dir / "data" / "trashcan_instance_raw")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data" / "trashcan_instance_yolo")
    return parser.parse_args()


def load_names(raw_dir):
    """Validate category IDs and return zero-indexed class names."""
    category_sets = []
    for split in ("train", "val"):
        with open(raw_dir / f"instances_{split}_trashcan.json", encoding="utf-8") as file:
            categories = sorted(json.load(file)["categories"], key=lambda category: category["id"])
        category_sets.append([(category["id"], category["name"]) for category in categories])
    if category_sets[0] != category_sets[1]:
        raise ValueError("Train and validation category definitions differ.")
    if [category_id for category_id, _ in category_sets[0]] != list(range(1, len(category_sets[0]) + 1)):
        raise ValueError("Category IDs must be contiguous and start at 1.")
    return [name for _, name in category_sets[0]]


def link_images(raw_dir, output_dir):
    """Hard-link images into the YOLO layout, copying only when linking is unavailable."""
    for split in ("train", "val"):
        image_dir = output_dir / "images" / split
        image_dir.mkdir(parents=True)
        for source in (raw_dir / split).glob("*.jpg"):
            target = image_dir / source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)


def main():
    """Convert the prepared TrashCan instance split into a trainable YOLO-Seg dataset."""
    args = parse_args()
    raw_dir, output_dir = args.raw_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    names = load_names(raw_dir)
    convert_coco(raw_dir, output_dir, use_segments=True, cls91to80=False)
    for split in ("train", "val"):
        (output_dir / "labels" / f"{split}_trashcan").rename(output_dir / "labels" / split)
    link_images(raw_dir, output_dir)
    yaml = "train: images/train\nval: images/val\nnames:\n" + "".join(
        f"  {index}: {json.dumps(name, ensure_ascii=False)}\n" for index, name in enumerate(names)
    )
    (output_dir / "trashcan_instance.yaml").write_text(yaml, encoding="utf-8")
    print(f"Converted {len(names)} classes to {output_dir}")


if __name__ == "__main__":
    main()
