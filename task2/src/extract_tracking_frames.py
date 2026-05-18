from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract consecutive annotated tracking frames for the report.")
    parser.add_argument("--video", default="runs/task2_track/tracked_traffic.mp4")
    parser.add_argument("--output", default="runs/task2_track/report_frames")
    parser.add_argument("--start-frame", type=int, default=None, help="First frame index to extract.")
    parser.add_argument("--start-sec", type=float, default=10.0, help="First timestamp if --start-frame is not set.")
    parser.add_argument("--num-frames", type=int, default=4, help="Number of consecutive frames to extract.")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        raise SystemExit(f"Cannot read FPS from video: {video_path}")

    if args.num_frames <= 0:
        raise SystemExit("--num-frames must be positive")
    start_frame = args.start_frame
    if start_frame is None:
        start_frame = int(round(args.start_sec * fps))
    if start_frame < 0:
        raise SystemExit("--start-frame/--start-sec must select a non-negative frame")

    saved = []
    for offset in range(args.num_frames):
        frame_index = start_frame + offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            print(f"skip frame {frame_index}: frame not available")
            continue
        sec = frame_index / fps
        out_path = output_dir / f"tracking_frame_{frame_index:06d}_{sec:.3f}s.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)

    cap.release()
    print(f"fps: {fps}")
    print(f"start_frame: {start_frame}")
    print(f"num_frames: {args.num_frames}")
    print("saved_frames:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
