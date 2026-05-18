#!/usr/bin/env bash
set -euo pipefail

python task2/src/train_yolov8.py \
  --preset full \
  --device "${YOLO_DEVICE:-2}" \
  --batch "${YOLO_BATCH:-4}" \
  --project runs/task2_train
