# 任务 2：YOLOv8 道路车辆检测与跟踪统计

本目录用于完成 PDF 中任务 2：使用 Road Vehicle Images Dataset 训练 YOLOv8 检测模型，并在 `traffic.mp4` 的 10-30 秒片段上完成视频流检测与多目标跟踪、遮挡/ID 跳变分析、越线计数。

## 目录说明

- `data/road_vehicle.yaml`：YOLOv8 数据集配置，默认指向 `archive/trafic_data`
- `src/validate_task2_assets.py`：检查数据集、标签分布和视频信息
- `src/train_yolov8.py`：GPU 服务器训练入口，支持小样本实验和正式实验
- `src/track_count.py`：截取视频片段，运行 YOLOv8 tracking，输出 bbox、类别、Tracking ID，并完成越线计数
- `src/summarize_tracking.py`：从 tracking CSV 重新汇总 ID 计数
- `src/extract_tracking_frames.py`：从遮挡或密集交汇片段中抽取连续 4 帧 tracking 截图用于报告
- `requirements.txt`：除 GPU Torch 外的依赖

## 1. 服务器环境

建议先在服务器上安装匹配 CUDA 的 GPU 版 PyTorch，然后安装本任务依赖：

```bash
python -m pip install -U pip
python -m pip install -r task2/requirements.txt
```

确认 GPU 可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 2. 检查数据

```bash
python task2/src/validate_task2_assets.py \
  --data task2/data/road_vehicle.yaml \
  --video traffic.mp4
```

当前本地检查结果：训练集 2704 张，验证集 300 张；`traffic.mp4` 为 4096x2160、12 FPS。

## 3. 小样本实验

小样本实验用于确认服务器环境、训练流程、loss/mAP 曲线是否正常。默认使用 `yolov8n.pt`、20 个 epoch、15% 训练数据、batch size 8：

```bash
python task2/src/train_yolov8.py \
  --preset small \
  --device 2 \
  --project runs/task2_train
```

训练脚本会自动把 `task2/data/road_vehicle.yaml` 转换成绝对路径版配置，保存到：

```text
runs/task2_train/_resolved_data/road_vehicle.yaml
```

如果服务器上的数据集不在默认的 `archive/trafic_data`，可以显式指定数据根目录：

```bash
python task2/src/train_yolov8.py \
  --preset small \
  --dataset-root /home/fuchenxi/hw2/archive/trafic_data \
  --device 2
```

如果只是测试代码能不能跑通，可以用更短的 smoke run：

```bash
python task2/src/train_yolov8.py \
  --preset smoke \
  --device 2
```

小样本输出通常位于：

```text
runs/task2_train/road_vehicle_small_yolov8n/
```

## 4. 正式实验

正式实验默认使用 `yolov8s.pt`、50 个 epoch、完整训练数据、batch size 4。服务器 GPU0 被占用时，脚本会通过 `CUDA_VISIBLE_DEVICES=2` 将物理 GPU2 隔离为进程内 GPU0，因此训练日志中显示 `CUDA:0` 是正常现象，实际使用的是物理 GPU2：

```bash
python task2/src/train_yolov8.py \
  --preset full \
  --device 2 \
  --project runs/task2_train
```

如果显存充足，可以提高 batch 或模型规模：

```bash
python task2/src/train_yolov8.py \
  --preset full \
  --model yolov8m.pt \
  --batch 4 \
  --imgsz 640 \
  --device 2 \
  --name road_vehicle_full_yolov8m
```

如果显存不够，可以降低设置：

```bash
python task2/src/train_yolov8.py \
  --preset full \
  --model yolov8n.pt \
  --batch 8 \
  --imgsz 512 \
  --device 2 \
  --name road_vehicle_full_yolov8n_512
```

断点续训：

```bash
python task2/src/train_yolov8.py \
  --preset full \
  --resume \
  --device 2
```

如果服务器显存仍然不稳定，可以进一步降低 batch：

```bash
python task2/src/train_yolov8.py \
  --preset full \
  --device 2 \
  --batch 2 \
  --workers 2 \
  --project runs/task2_train
```

如果看到类似 `images not found, missing path '/home/fuchenxi/archive/trafic_data/valid/images'` 的错误，说明数据集路径被按错误工作目录解析了。请更新到当前版本脚本后重新运行，或加上：

```bash
--dataset-root /home/fuchenxi/hw2/archive/trafic_data
```

正式权重通常位于：

```text
runs/task2_train/road_vehicle_full_yolov8s/weights/best.pt
```

## 5. 视频流检测、多目标跟踪与越线计数

用训练好的 `best.pt` 对交通视频截取 1 分 50 秒到 2 分 10 秒的 20 秒片段，并输出带 bbox、类别、Tracking ID、虚拟计数线和越线数量的视频。`--line` 可以使用像素坐标 `x1,y1,x2,y2`，也可以使用 0-1 归一化坐标。本次实验使用 `0.1,0.40,0.9,0.40`，即画面 40% 高度处的水平线。

```bash
python task2/src/track_count.py \
  --weights runs/task2_train/road_vehicle_full_yolov8s/weights/best.pt \
  --video traffic.mp4 \
  --start-sec 110 \
  --end-sec 130 \
  --resize-width 1280 \
  --line 0.1,0.40,0.9,0.40 \
  --device 2 \
  --output runs/task2_track
```

示例：设置一条位于画面 55% 高度的水平线：

```bash
python task2/src/track_count.py \
  --weights runs/task2_train/road_vehicle_full_yolov8s/weights/best.pt \
  --video traffic.mp4 \
  --start-sec 110 \
  --end-sec 130 \
  --resize-width 1280 \
  --line 0.1,0.40,0.9,0.40 \
  --device 2 \
  --output runs/task2_track
```

输出文件：

- `runs/task2_track/traffic_clip.mp4`：用于实验的 20 秒片段
- `runs/task2_track/tracked_traffic.mp4`：带 bbox、类别、Tracking ID、中心点、计数线和越线总数的结果视频
- `runs/task2_track/tracking_results.csv`：逐帧 bbox、中心点、类别、置信度和 Tracking ID
- `runs/task2_track/line_crossing_events.csv`：每个首次跨线 Tracking ID 的越线帧、方向和中心点坐标
- `runs/task2_track/summary.json`：唯一 Tracking ID 总数、越线总数和各类别数量

抽取连续 4 帧报告截图。建议先在结果视频中找到遮挡或车辆密集交汇片段，再用 `--start-sec` 或 `--start-frame` 指定连续帧起点：

```bash
python task2/src/extract_tracking_frames.py \
  --video runs/task2_track/tracked_traffic.mp4 \
  --start-sec 10 \
  --num-frames 4
```