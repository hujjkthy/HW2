from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_dir(data_yaml: Path, cfg: dict) -> Path:
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def count_images(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def count_label_classes(labels_dir: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            counts[int(float(parts[0]))] += 1
    return counts


def inspect_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    cap.release()
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Task 2 dataset and traffic video assets.")
    parser.add_argument("--data", default="task2/data/road_vehicle.yaml", help="YOLO dataset YAML path.")
    parser.add_argument("--video", default="traffic.mp4", help="Traffic video path.")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    video_path = Path(args.video).resolve()
    cfg = read_yaml(data_yaml)
    dataset_dir = resolve_dataset_dir(data_yaml, cfg)

    train_images = dataset_dir / cfg["train"]
    valid_images = dataset_dir / cfg["val"]
    train_labels = dataset_dir / "train" / "labels"
    valid_labels = dataset_dir / "valid" / "labels"

    print(f"data_yaml: {data_yaml}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"classes: {cfg['nc']} -> {', '.join(cfg['names'])}")
    print(f"train_images: {train_images} ({count_images(train_images)} images)")
    print(f"valid_images: {valid_images} ({count_images(valid_images)} images)")
    print(f"train_labels: {train_labels} ({len(list(train_labels.glob('*.txt')))} files)")
    print(f"valid_labels: {valid_labels} ({len(list(valid_labels.glob('*.txt')))} files)")

    train_counts = count_label_classes(train_labels)
    valid_counts = count_label_classes(valid_labels)
    print("label_instances_by_class:")
    for class_id, name in enumerate(cfg["names"]):
        print(f"  {class_id:02d} {name}: train={train_counts[class_id]}, valid={valid_counts[class_id]}")

    video = inspect_video(video_path)
    print("video:")
    for key, value in video.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
