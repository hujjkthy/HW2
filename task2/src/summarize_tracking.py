from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize YOLOv8 tracking CSV into ID-level vehicle counts.")
    parser.add_argument("--csv", default="runs/task2_track/tracking_results.csv")
    parser.add_argument("--output", default="runs/task2_track/id_summary.json")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    by_id: dict[int, dict] = {}
    class_votes: dict[int, Counter[str]] = defaultdict(Counter)
    frames_by_id: dict[int, set[int]] = defaultdict(set)

    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            track_id = int(row["track_id"])
            frame = int(row["frame"])
            class_votes[track_id][row["class_name"]] += 1
            frames_by_id[track_id].add(frame)
            if track_id not in by_id:
                by_id[track_id] = {
                    "track_id": track_id,
                    "first_frame": frame,
                    "last_frame": frame,
                    "detections": 0,
                }
            by_id[track_id]["first_frame"] = min(by_id[track_id]["first_frame"], frame)
            by_id[track_id]["last_frame"] = max(by_id[track_id]["last_frame"], frame)
            by_id[track_id]["detections"] += 1

    class_counts: Counter[str] = Counter()
    ids = []
    for track_id in sorted(by_id):
        top_class = class_votes[track_id].most_common(1)[0][0]
        class_counts[top_class] += 1
        item = by_id[track_id]
        item["class_name"] = top_class
        item["observed_frames"] = len(frames_by_id[track_id])
        ids.append(item)

    summary = {
        "total_unique_track_ids": len(ids),
        "counts_by_class": dict(sorted(class_counts.items())),
        "tracks": ids,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
