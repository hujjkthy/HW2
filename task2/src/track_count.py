from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml


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


def load_class_names(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return list(cfg["names"])


def parse_line(line: str, width: int, height: int) -> tuple[float, float, float, float]:
    if line == "auto":
        return (width * 0.5, height * 0.15, width * 0.5, height * 0.95)
    parts = [float(item.strip()) for item in line.split(",")]
    if len(parts) != 4:
        raise SystemExit("--line must be 'auto' or four comma-separated values: x1,y1,x2,y2")
    if all(0.0 <= item <= 1.0 for item in parts):
        x1, y1, x2, y2 = parts
        return (x1 * width, y1 * height, x2 * width, y2 * height)
    return tuple(parts)  # type: ignore[return-value]


def line_side(point: tuple[float, float], line: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def crosses_line(
    previous: tuple[float, float],
    current: tuple[float, float],
    line: tuple[float, float, float, float],
) -> tuple[bool, str]:
    prev_side = line_side(previous, line)
    curr_side = line_side(current, line)
    if prev_side == 0 or curr_side == 0:
        return False, "touch"
    if prev_side * curr_side < 0:
        direction = "negative_to_positive" if prev_side < curr_side else "positive_to_negative"
        return True, direction
    return False, "none"


def make_clip(
    source: Path,
    target: Path,
    start_sec: float,
    duration_sec: float,
    resize_width: int | None,
) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(round(start_sec * fps)))
    requested_frames = int(round(duration_sec * fps))
    end_frame = min(total_frames, start_frame + requested_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_width, out_height = width, height
    if resize_width and resize_width > 0 and resize_width < width:
        scale = resize_width / width
        out_width = resize_width
        out_height = int(round(height * scale))

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )

    written = 0
    while start_frame + written < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (out_width, out_height) != (width, height):
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    return {
        "source_fps": fps,
        "source_width": width,
        "source_height": height,
        "clip_width": out_width,
        "clip_height": out_height,
        "start_sec": start_sec,
        "duration_sec": written / fps if fps else 0.0,
        "frames": written,
        "clip_path": str(target),
    }


def draw_tracks(frame, rows: list[dict], class_names: list[str]) -> None:
    for row in rows:
        x1, y1, x2, y2 = (int(row[k]) for k in ("x1", "y1", "x2", "y2"))
        track_id = row["track_id"]
        cls_id = int(row["class_id"])
        conf = float(row["confidence"])
        label = f"ID {track_id} {class_names[cls_id]} {conf:.2f}"
        color = ((37 * (track_id + 3)) % 255, (17 * (track_id + 7)) % 255, (29 * (track_id + 11)) % 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (int(row["center_x"]), int(row["center_y"])), 4, color, -1)


def draw_counting_line(
    frame,
    line: tuple[float, float, float, float],
    count: int,
    recent_crossing_ids: list[int],
) -> None:
    x1, y1, x2, y2 = (int(v) for v in line)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(frame, "Counting line", (x1 + 8, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    text = f"Crossing count: {count}"
    cv2.rectangle(frame, (16, 16), (360, 72), (0, 0, 0), -1)
    cv2.putText(frame, text, (28, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if recent_crossing_ids:
        ids_text = "Latest IDs: " + ", ".join(str(i) for i in recent_crossing_ids[-5:])
        cv2.putText(frame, ids_text, (28, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def track_and_count(args: argparse.Namespace) -> None:
    YOLO = require_ultralytics()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data).resolve()
    class_names = load_class_names(data_yaml)
    video_path = Path(args.video).resolve()
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    with tempfile.TemporaryDirectory(prefix="task2_clip_") as tmp:
        clip_path = Path(tmp) / "traffic_clip.mp4"
        clip_info = make_clip(video_path, clip_path, args.start_sec, args.duration_sec, args.resize_width)
        saved_clip = output_dir / "traffic_clip.mp4"
        shutil.copy2(clip_path, saved_clip)
        clip_info["clip_path"] = str(saved_clip)

        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        counting_line = parse_line(args.line, width, height)
        writer = cv2.VideoWriter(
            str(output_dir / "tracked_traffic.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        model = YOLO(str(weights_path))
        results = model.track(
            source=str(clip_path),
            stream=True,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=choose_device(args.device),
            verbose=False,
        )

        unique_ids: set[int] = set()
        class_by_id: dict[int, Counter[str]] = defaultdict(Counter)
        per_frame_rows: list[dict] = []
        previous_center_by_id: dict[int, tuple[float, float]] = {}
        counted_track_ids: set[int] = set()
        crossing_events: list[dict] = []
        recent_crossing_ids: list[int] = []

        frame_index = 0
        for result in results:
            frame = result.orig_img.copy()
            frame_rows: list[dict] = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int)
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                for box, track_id, cls_id, conf in zip(xyxy, track_ids, cls_ids, confs):
                    center_x = float((box[0] + box[2]) / 2.0)
                    center_y = float((box[1] + box[3]) / 2.0)
                    row = {
                        "frame": frame_index,
                        "time_sec": frame_index / fps if fps else 0.0,
                        "track_id": int(track_id),
                        "class_id": int(cls_id),
                        "class_name": class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id),
                        "confidence": float(conf),
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                    unique_ids.add(int(track_id))
                    class_by_id[int(track_id)][row["class_name"]] += 1
                    current_center = (center_x, center_y)
                    previous_center = previous_center_by_id.get(int(track_id))
                    if previous_center is not None and int(track_id) not in counted_track_ids:
                        crossed, direction = crosses_line(previous_center, current_center, counting_line)
                        if crossed:
                            counted_track_ids.add(int(track_id))
                            recent_crossing_ids.append(int(track_id))
                            crossing_events.append(
                                {
                                    "frame": frame_index,
                                    "time_sec": frame_index / fps if fps else 0.0,
                                    "track_id": int(track_id),
                                    "class_id": int(cls_id),
                                    "class_name": row["class_name"],
                                    "direction": direction,
                                    "prev_center_x": previous_center[0],
                                    "prev_center_y": previous_center[1],
                                    "center_x": center_x,
                                    "center_y": center_y,
                                }
                            )
                    previous_center_by_id[int(track_id)] = current_center
                    frame_rows.append(row)
                    per_frame_rows.append(row)
            draw_tracks(frame, frame_rows, class_names)
            draw_counting_line(frame, counting_line, len(counted_track_ids), recent_crossing_ids)
            writer.write(frame)
            frame_index += 1

        writer.release()

    csv_path = output_dir / "tracking_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame",
            "time_sec",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(per_frame_rows)

    crossing_csv_path = output_dir / "line_crossing_events.csv"
    with crossing_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame",
            "time_sec",
            "track_id",
            "class_id",
            "class_name",
            "direction",
            "prev_center_x",
            "prev_center_y",
            "center_x",
            "center_y",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(crossing_events)

    final_class_counts: Counter[str] = Counter()
    for counter in class_by_id.values():
        if counter:
            final_class_counts[counter.most_common(1)[0][0]] += 1
    crossing_class_counts: Counter[str] = Counter(event["class_name"] for event in crossing_events)

    summary = {
        "weights": str(weights_path),
        "video": str(video_path),
        "clip": clip_info,
        "tracker": args.tracker,
        "confidence": args.conf,
        "iou": args.iou,
        "frames_processed": frame_index,
        "total_unique_track_ids": len(unique_ids),
        "counts_by_class": dict(sorted(final_class_counts.items())),
        "counting_line": {
            "x1": counting_line[0],
            "y1": counting_line[1],
            "x2": counting_line[2],
            "y2": counting_line[3],
        },
        "total_line_crossing_count": len(counted_track_ids),
        "line_crossing_counts_by_class": dict(sorted(crossing_class_counts.items())),
        "outputs": {
            "annotated_video": str(output_dir / "tracked_traffic.mp4"),
            "tracking_csv": str(csv_path),
            "line_crossing_csv": str(crossing_csv_path),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 tracking and count line-crossing objects.")
    parser.add_argument("--weights", required=True, help="Trained YOLOv8 weights, usually runs/.../weights/best.pt.")
    parser.add_argument("--video", default="traffic.mp4")
    parser.add_argument("--data", default="task2/data/road_vehicle.yaml")
    parser.add_argument("--output", default="runs/task2_track")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None, help="Optional clip end time in seconds.")
    parser.add_argument("--duration-sec", type=float, default=20.0, help="Use 10-30 seconds for the homework sample.")
    parser.add_argument("--resize-width", type=int, default=1280, help="Downscale 4K video for faster tracking. Use 0 to keep original.")
    parser.add_argument(
        "--line",
        default="auto",
        help="Counting line as x1,y1,x2,y2 in pixels or normalized 0-1 values. Default: vertical center line.",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, 2, 0,1, etc. Use --device 2 for GPU2.")
    args = parser.parse_args()
    if args.end_sec is not None:
        if args.end_sec <= args.start_sec:
            raise SystemExit("--end-sec must be greater than --start-sec")
        args.duration_sec = args.end_sec - args.start_sec
    if not 10 <= args.duration_sec <= 30:
        print("Warning: homework asks for a 10-30 second video sample.")
    if args.resize_width <= 0:
        args.resize_width = None
    track_and_count(args)


if __name__ == "__main__":
    main()
