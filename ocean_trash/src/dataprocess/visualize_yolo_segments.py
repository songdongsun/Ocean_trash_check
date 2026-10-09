# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from ultralytics.utils.plotting import colors

CHINESE_NAMES = {
    0: "水下机器人",
    1: "植物",
    2: "鱼类",
    3: "海星",
    4: "贝类",
    5: "螃蟹",
    6: "鳗鱼",
    7: "其他动物",
    8: "衣物垃圾",
    9: "管道垃圾",
    10: "瓶子",
    11: "袋子",
    12: "零食包装袋",
    13: "罐子",
    14: "杯子",
    15: "容器",
    16: "未知垃圾",
    17: "树枝",
    18: "残骸",
    19: "防水布",
    20: "绳索",
    21: "渔网",
}


def parse_args():
    """Parse visualization options."""
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Review YOLO segmentation labels overlaid on their source images.")
    parser.add_argument("--dataset-dir", type=Path, default=project_dir / "data" / "trashcan_instance_yolo")
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--start", type=int, default=0, help="Zero-based index at which to start reviewing.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum images to review; 0 means all remaining images.")
    parser.add_argument("--save-dir", type=Path, help="Batch-save overlays instead of opening the review window.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Mask opacity from 0 to 1.")
    return parser.parse_args()


def load_names(dataset_dir):
    """Load class names from the generated dataset YAML."""
    with open(dataset_dir / "trashcan_instance.yaml", encoding="utf-8") as file:
        names = yaml.safe_load(file)["names"]
    return {int(index): name for index, name in names.items()}


def draw_labels(image_path, dataset_dir, names, alpha):
    """Draw filled segmentation polygons, outlines, and class names on one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    overlay = image.copy()
    height, width = image.shape[:2]
    label_path = dataset_dir / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
    annotations = []
    if label_path.exists():
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            values = np.asarray(line.split(), dtype=np.float32)
            if len(values) < 7 or len(values) % 2 == 0:
                raise ValueError(f"Invalid segmentation row: {label_path}:{line_number}")
            class_id = int(values[0])
            points = values[1:].reshape(-1, 2) * (width, height)
            points = np.clip(np.rint(points), (0, 0), (width - 1, height - 1)).astype(np.int32)
            annotations.append((class_id, points))
            cv2.fillPoly(overlay, [points], colors(class_id, True))
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    for class_id, points in annotations:
        color = colors(class_id, True)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
    for class_id, points in annotations:
        color = colors(class_id, True)
        x, y = points[np.argmin(points[:, 1])]
        label = f"{class_id}: {CHINESE_NAMES[class_id]} ({names[class_id]})"
        box = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = box[2] - box[0], box[3] - box[1]
        x = max(0, min(int(x), width - text_width - 6))
        y = max(0, min(int(y) - text_height - 6, height - text_height - 6))
        draw.rectangle((x, y, x + text_width + 6, y + text_height + 6), fill=color[::-1])
        draw.text((x + 3, y + 1), label, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), len(annotations)


def image_paths(dataset_dir, split):
    """Return images in deterministic review order."""
    splits = ("train", "val") if split == "both" else (split,)
    return [path for name in splits for path in sorted((dataset_dir / "images" / name).glob("*.jpg"))]


def save_image(image, image_path, save_dir):
    """Save an overlay while retaining its train/val subdirectory."""
    target = save_dir / image_path.parent.name / image_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Unable to write image: {target}")
    return target


def main():
    """Interactively review labels or export annotated images."""
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not 0 <= args.alpha <= 1 or args.start < 0 or args.limit < 0:
        raise ValueError("alpha must be within 0..1, and start/limit must be non-negative.")
    paths = image_paths(dataset_dir, args.split)[args.start :]
    paths = paths[: args.limit or None]
    if not paths:
        raise FileNotFoundError("No images matched the requested dataset split and range.")
    names = load_names(dataset_dir)

    if args.save_dir:
        save_dir = args.save_dir.resolve()
        for index, path in enumerate(paths, 1):
            image, count = draw_labels(path, dataset_dir, names, args.alpha)
            save_image(image, path, save_dir)
            print(f"[{index}/{len(paths)}] {path.name}: {count} objects")
        print(f"Saved {len(paths)} annotated images to {save_dir}")
        return

    index = 0
    print("Keys: D/right/space = next, A/left = previous, S = save current, Q/Esc = quit")
    while True:
        path = paths[index]
        image, count = draw_labels(path, dataset_dir, names, args.alpha)
        title = f"TrashCan label review [{index + 1}/{len(paths)}] {path.parent.name}/{path.name} ({count} objects)"
        scale = min(1600 / image.shape[1], 900 / image.shape[0], 1.0)
        display = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        cv2.imshow(title, display)
        key = cv2.waitKeyEx(0)
        cv2.destroyWindow(title)
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("a"), ord("A"), 2424832):
            index = max(0, index - 1)
        elif key in (ord("s"), ord("S")):
            target = save_image(image, path, dataset_dir / "reviewed")
            print(f"Saved {target}")
        else:
            index = min(len(paths) - 1, index + 1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
