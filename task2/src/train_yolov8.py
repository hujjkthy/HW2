from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


PRESETS = {
    "smoke": {
        "model": "yolov8n.pt",
        "epochs": 1,
        "imgsz": 416,
        "batch": -1,
        "workers": 4,
        "fraction": 0.02,
        "name": "road_vehicle_smoke_yolov8n",
        "description": "Very short GPU/server sanity check.",
    },
    "small": {
        "model": "yolov8n.pt",
        "epochs": 20,
        "imgsz": 640,
        "batch": 8,
        "workers": 4,
        "fraction": 0.15,
        "name": "road_vehicle_small_yolov8n",
        "description": "Small-sample experiment for quick comparison and debugging.",
    },
    "full": {
        "model": "yolov8s.pt",
        "epochs": 50,
        "imgsz": 640,
        "batch": 4,
        "workers": 4,
        "fraction": 1.0,
        "name": "road_vehicle_full_yolov8s",
        "description": "Formal experiment on the full dataset.",
    },
}


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install it with:\n"
            "  python -m pip install -r task2/requirements.txt"
        ) from exc
    return YOLO


def choose_device(requested: str) -> str:
    requested = os.environ.get("YOLO_DEVICE") if requested == "auto" and os.environ.get("YOLO_DEVICE") else requested
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if requested != "auto":
        if requested.isdigit() and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = requested
            return "0"
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def print_runtime(device: str) -> None:
    try:
        import torch
    except ImportError:
        print("torch: not installed")
        return

    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "selected_device": device,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def preset_value(args: argparse.Namespace, key: str):
    value = getattr(args, key)
    if value is not None:
        return value
    if args.preset == "custom":
        return None
    return PRESETS[args.preset][key]


def resolve_dataset_root(data_yaml: Path, cfg: dict, override: str | None = None) -> Path:
    if override:
        root = Path(override).expanduser().resolve()
        if not root.exists():
            raise SystemExit(f"Dataset root not found: {root}")
        return root

    raw_root = Path(cfg.get("path", ".")).expanduser()
    candidates = []
    if raw_root.is_absolute():
        candidates.append(raw_root)
    else:
        candidates.extend(
            [
                data_yaml.parent / raw_root,
                Path.cwd() / raw_root,
                data_yaml.parents[2] / raw_root if len(data_yaml.parents) > 2 else data_yaml.parent / raw_root,
            ]
        )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    checked = "\n  ".join(str(p.resolve()) for p in candidates)
    raise SystemExit(
        "Dataset root not found. Checked:\n"
        f"  {checked}\n"
        "Pass the correct path with --dataset-root, for example:\n"
        "  python task2/src/train_yolov8.py --preset small --dataset-root /home/fuchenxi/hw2/archive/trafic_data"
    )


def write_resolved_data_yaml(data_yaml: Path, output_dir: Path, dataset_root: Path) -> Path:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_images = dataset_root / cfg["train"]
    val_images = dataset_root / cfg["val"]
    missing = [str(path) for path in (train_images, val_images) if not path.exists()]
    if missing:
        raise SystemExit(
            "Dataset image folders not found:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\nCheck whether archive/trafic_data was copied to the server."
        )

    resolved_cfg = dict(cfg)
    resolved_cfg["path"] = str(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = output_dir / data_yaml.name
    with resolved_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=False, allow_unicode=True)
    return resolved_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the Road Vehicle Images dataset.")
    parser.add_argument(
        "--preset",
        choices=["smoke", "small", "full", "custom"],
        default="custom",
        help="Experiment preset. Use small for quick experiments and full for the formal run.",
    )
    parser.add_argument("--data", default="task2/data/road_vehicle.yaml", help="Dataset YAML path.")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Absolute dataset root containing train/ and valid/. Overrides the YAML path field.",
    )
    parser.add_argument("--model", default=None, help="YOLOv8 checkpoint, e.g. yolov8n.pt/yolov8s.pt.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None, help="Use -1 for Ultralytics auto batch on GPU.")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, 2, 0,1, etc. Use --device 2 for GPU2.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None, help="Fraction of training data. small preset uses 0.15.")
    parser.add_argument("--cache", choices=["none", "ram", "disk"], default="none")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Mixed precision training.")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate. Leave unset for YOLO default.")
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Resume the last run under the same project/name.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project", default="runs/task2_train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    if args.preset != "custom":
        print(f"Using preset: {args.preset} ({PRESETS[args.preset]['description']})")

    model_name = preset_value(args, "model") or "yolov8n.pt"
    epochs = preset_value(args, "epochs") or 100
    imgsz = preset_value(args, "imgsz") or 640
    batch = preset_value(args, "batch")
    if batch is None:
        batch = 16
    workers = preset_value(args, "workers")
    if workers is None:
        workers = 8
    fraction = preset_value(args, "fraction")
    if fraction is None:
        fraction = 1.0
    name = preset_value(args, "name") or f"road_vehicle_{args.preset}_{Path(model_name).stem}"
    device = choose_device(args.device)
    cache = False if args.cache == "none" else args.cache
    project = Path(args.project).expanduser().resolve()
    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = resolve_dataset_root(data_path, data_cfg, args.dataset_root)
    resolved_data_path = write_resolved_data_yaml(data_path, project / "_resolved_data", dataset_root)

    print_runtime(device)

    YOLO = require_ultralytics()
    model = YOLO(model_name)
    train_kwargs = {
        "data": str(resolved_data_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "project": str(project),
        "name": name,
        "seed": args.seed,
        "patience": args.patience,
        "exist_ok": args.exist_ok,
        "plots": True,
        "save": True,
        "val": True,
        "cache": cache,
        "amp": args.amp,
        "optimizer": args.optimizer,
        "cos_lr": args.cos_lr,
        "resume": args.resume,
        "pretrained": args.pretrained,
        "fraction": fraction,
    }
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0

    print("Training configuration:")
    print(json.dumps(train_kwargs, ensure_ascii=False, indent=2, default=str))
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
