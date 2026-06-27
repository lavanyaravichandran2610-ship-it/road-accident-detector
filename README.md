# 🚨 Road Accident Detection System

> Automatically detect road accidents from CCTV / traffic-camera footage using
> YOLOv8 + OpenCV, and generate immediate alerts with screenshot and video-clip evidence.

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Modules](#modules)
6. [Detection Logic](#detection-logic)
7. [Dashboard](#dashboard)
8. [Dataset Sources](#dataset-sources)
9. [Configuration Reference](#configuration-reference)
10. [Running Tests](#running-tests)
11. [Extending the System](#extending-the-system)

---

## Overview

Emergency services are often notified of accidents several minutes after they
occur — critical time that can determine survival outcomes.

This system continuously monitors road-camera feeds and:

- Detects vehicles in every frame using **YOLOv8**
- Tracks each vehicle's position, speed, and shape across frames
- Flags accidents based on **collision**, **sudden stop**, and **rollover** signals
- Saves annotated **screenshots** and short **video clips** as evidence
- Logs every event to a **JSON file** for downstream reporting
- (Optionally) fires an **HTTP webhook** for integration with dispatch systems
- Presents a **Streamlit dashboard** for live monitoring

---

## Architecture

```
Camera Feed (CCTV / File / Webcam / RTSP)
         │
         ▼
┌────────────────────┐
│  Module 1          │  VideoInputHandler
│  Video Input       │  – frame-by-frame reading
│                    │  – resize & skip-frame control
└────────┬───────────┘
         │  frame (numpy array)
         ▼
┌────────────────────┐
│  Module 2          │  VehicleDetector (YOLOv8 + ByteTrack)
│  Vehicle Detection │  – detect cars, bikes, buses, trucks
│                    │  – maintain per-vehicle speed & position history
└────────┬───────────┘
         │  List[Vehicle]
         ▼
┌────────────────────┐
│  Module 3          │  AccidentDetector
│  Accident Detection│  – IoU collision check
│                    │  – sudden-speed-drop check
│                    │  – aspect-ratio rollover check
└────────┬───────────┘
         │  List[AccidentEvent]
         ▼
┌────────────────────┐
│  Module 4          │  AlertManager
│  Alert Generation  │  – screenshot save
│                    │  – clip save (pre + post buffer)
│                    │  – JSON event log
│                    │  – HTTP webhook (optional)
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Module 5          │  Streamlit Dashboard
│  Dashboard         │  – live annotated video
│                    │  – metrics, event log, gallery
└────────────────────┘
```

---

## Project Structure

```
road_accident_detection/
│
├── main.py                        # CLI entry point
├── requirements.txt
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── video_input.py             # Module 1 – frame reading
│   ├── vehicle_detection.py       # Module 2 – YOLOv8 + tracking
│   ├── accident_detection.py      # Module 3 – accident logic
│   └── alert_system.py           # Module 4 – alerts & storage
│
├── dashboard/
│   └── app.py                    # Module 5 – Streamlit UI
│
├── utils/
│   └── dataset_helper.py         # Kaggle / YouTube dataset tools
│
├── tests/
│   └── test_modules.py           # 19 unit tests (all pass)
│
├── alerts/                       # Created at runtime
│   ├── screenshots/
│   ├── clips/
│   └── event_log.json
│
└── models/                       # Place custom .pt weights here
```

---

## Quick Start

### 1 — Install dependencies

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

> **GPU users:** replace `torch` with the CUDA build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2 — Run on a video file

```bash
python main.py --source road_video.mp4 --show
```

### 3 — Launch the web dashboard

```bash
python main.py --dashboard
# or directly:
streamlit run dashboard/app.py
```

### 4 — Use a webcam

```bash
python main.py --source 0 --show
```

### 5 — RTSP / CCTV stream

```bash
python main.py --source rtsp://192.168.1.100:554/live/main --headless
```

---

## Modules

### Module 1 — Video Input (`modules/video_input.py`)

| Class / Function | Purpose |
|---|---|
| `VideoInputHandler` | Context manager that reads frames from any source |
| `.stream_frames()` | Generator yielding `(frame, metadata)` pairs |
| `.seek(n)` | Jump to frame `n` (file sources) |
| `extract_thumbnail(path)` | Pull a single frame from a video file |
| `list_available_cameras()` | Detect connected webcam indices |

**Metadata dict** produced per frame:
```python
{
  "frame_idx": 142,
  "timestamp_s": 4.733,
  "source_fps": 30.0,
  "width": 1280,
  "height": 720,
  "source": "road_video.mp4",
}
```

---

### Module 2 — Vehicle Detection (`modules/vehicle_detection.py`)

Uses **YOLOv8** with integrated **ByteTrack** tracking (via `ultralytics`).

**Detected classes:** `car`, `motorcycle`, `bus`, `truck`

| Class / Function | Purpose |
|---|---|
| `VehicleDetector` | Wraps YOLOv8; maintains vehicle registry |
| `.detect(frame, frame_idx)` | Returns `List[Vehicle]` |
| `.draw(frame, vehicles)` | Returns annotated frame |
| `Vehicle` | Dataclass tracking bbox, positions, speeds |
| `Vehicle.speed_dropped_suddenly()` | True if recent speed < 20 % of rolling avg |

**Choosing a model:**

| Model | Size | Speed | Accuracy | Use case |
|---|---|---|---|---|
| `yolov8n.pt` | 6 MB | ★★★★★ | ★★★ | Raspberry Pi / low-end hardware |
| `yolov8s.pt` | 22 MB | ★★★★ | ★★★★ | Balanced (recommended default) |
| `yolov8m.pt` | 52 MB | ★★★ | ★★★★★ | Server with dedicated GPU |

---

### Module 3 — Accident Detection (`modules/accident_detection.py`)

Three independent detectors run on every frame:

#### 3a — Collision Detection (IoU-based)
```
For every pair of vehicles:
  Compute Intersection-over-Union of their bounding boxes.
  If IoU > threshold (default 0.15):
    Fire COLLISION event
    Confidence = f(IoU, combined speed)
```

#### 3b — Sudden Stop Detection (speed history)
```
For each vehicle with ≥ 8 frames of history:
  history_avg = mean(speeds[:-3])      # rolling average
  recent_avg  = mean(speeds[-3:])      # last 3 frames
  If history_avg > min_speed AND recent_avg < history_avg × 0.20:
    Fire SUDDEN_STOP event
```

#### 3c — Rollover Detection (aspect-ratio shift)
```
For each vehicle:
  aspect = bbox_width / bbox_height
  If baseline aspect > 0.5 (landscape = normal vehicle):
    If |current_aspect - baseline| / baseline > 0.55:
      Fire ROLLOVER event
```

A **cooldown** (default 45 frames) prevents the same accident from generating
duplicate alerts on consecutive frames.

---

### Module 4 — Alert Generation (`modules/alert_system.py`)

Every confirmed accident triggers:

1. **Console log** — timestamped warning with event details
2. **Screenshot** — JPEG with metadata burned in, saved to `alerts/screenshots/`
3. **Video clip** — MP4 with 5 s before and 5 s after the event (pre-event frames
   are buffered continuously; post-event frames are collected after trigger)
4. **JSON log** — `alerts/event_log.json` appended with every event record
5. **Webhook** (optional) — HTTP POST to your configured endpoint

JSON event record shape:
```json
{
  "type": "COLLISION",
  "label": "Vehicle Collision",
  "frame_idx": 420,
  "timestamp_s": 14.0,
  "confidence": 0.87,
  "vehicle_ids": [3, 7],
  "location_px": [640, 360],
  "notes": "IoU=0.31 | speeds 12.4/9.8 px/fr",
  "event_id": "COLLISION_20240615_143022",
  "screenshot": "alerts/screenshots/COLLISION_20240615_143022.jpg"
}
```

---

### Module 5 — Dashboard (`dashboard/app.py`)

Run with:
```bash
streamlit run dashboard/app.py
```

Features:
- **Sidebar** — configure source, model, thresholds, alert paths
- **Live feed** — annotated video stream with bounding boxes and accident banners
- **Metrics row** — accident count, frames processed, processing FPS, status
- **Event log** — scrollable reverse-chronological accident log with confidence badges
- **Screenshot gallery** — latest 9 accident screenshots shown inline

---

## Detection Logic

### Why IoU for collisions?

When two vehicles actually collide, their bounding boxes physically overlap in the
image. IoU (Intersection-over-Union) quantifies that overlap as a ratio from 0
(no overlap) to 1 (identical boxes). A threshold of 0.15 means the boxes must
share at least 15 % of their combined area — enough to catch real impacts while
ignoring vehicles that happen to be in the same lane.

### Why speed history for sudden stops?

A vehicle braking normally decelerates gradually. After an impact, the speed
drops to near-zero in 1–3 frames. By comparing the rolling average against the
most recent 3 frames, the detector catches this discontinuity without triggering
on normal slow-downs.

### Why aspect ratio for rollovers?

A standard car on the road shows a wide, low bounding box (landscape, ratio > 1).
A rolled/flipped vehicle shows a tall, narrow box (portrait, ratio < 1). Tracking
the ratio change over time (rather than a single frame) reduces false positives
from camera perspective changes.

---

## Dataset Sources

### Kaggle (recommended)

```bash
# Install kaggle CLI and set up ~/.kaggle/kaggle.json first
python utils/dataset_helper.py download-kaggle road-accident-detection
python utils/dataset_helper.py download-kaggle car-crash-dataset
python utils/dataset_helper.py list   # see all available
```

**Top Kaggle datasets:**
- `ckay16/accident-detection-from-cctv-footage` — CCTV footage, labelled frames
- `asefjamilajwad/car-crash-dataset` — dashcam images with bounding boxes
- `hendrickemanuel/road-accidents-dataset` — multi-class accident images

### YouTube test videos

```bash
python utils/dataset_helper.py download-youtube "https://youtube.com/watch?v=..."
```

### Local scan

```bash
python utils/dataset_helper.py scan ./my_footage/
```

---

## Configuration Reference

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file, webcam index, or RTSP URL |
| `--model` | `yolov8n.pt` | YOLOv8 weights (auto-downloaded on first run) |
| `--conf` | `0.40` | YOLO detection confidence threshold |
| `--iou` | `0.15` | Collision IoU threshold |
| `--speed-drop` | `0.20` | Sudden-stop sensitivity (lower = more sensitive) |
| `--cooldown` | `45` | Min frames between same alert type |
| `--device` | `cpu` | Inference device (`cpu` or `cuda`) |
| `--skip` | `2` | Process every Nth frame (1 = every frame) |
| `--alerts-dir` | `alerts` | Output directory for screenshots & clips |
| `--clip-pre` | `5.0` | Seconds of footage before accident in clip |
| `--clip-post` | `5.0` | Seconds of footage after accident in clip |
| `--webhook` | `` | HTTP POST URL for real-time notifications |
| `--output` | `` | Path to save annotated output video |
| `--headless` | off | Disable display window (for servers) |
| `--max-seconds` | `0` | Stop after N seconds (0 = unlimited) |
| `--dashboard` | off | Launch Streamlit instead of CLI pipeline |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Expected output:
```
19 passed in 0.56s
```

Tests cover:
- Source-type detection (file / webcam / RTSP)
- Vehicle centre and area calculations
- Speed-drop detection logic
- IoU computation (identical boxes = 1.0; no overlap = 0.0)
- Collision event firing and cooldown enforcement
- Sudden-stop triggering
- Screenshot saving and JSON log persistence

---

## Extending the System

### Add a new accident type

1. Add a member to `AccidentType` in `accident_detection.py`
2. Add an entry to `ACCIDENT_LABELS` and `ACCIDENT_COLORS`
3. Write a `_check_<name>` method in `AccidentDetector`
4. Call it in `AccidentDetector.analyse()`
5. Add tests to `tests/test_modules.py`

### Custom YOLOv8 model (fine-tuned on accident data)

```bash
# Create dataset.yaml
python utils/dataset_helper.py scan ./accident_dataset/

# Train
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(data="dataset.yaml", epochs=50, imgsz=640)

# Use the fine-tuned weights
python main.py --source video.mp4 --model runs/detect/train/weights/best.pt
```

### Webhook integration example (Node.js receiver)

```javascript
const express = require("express");
const app = express();
app.use(express.json());
app.post("/accident-webhook", (req, res) => {
  console.log("Accident detected:", req.body);
  // → notify dispatch, update database, send SMS, etc.
  res.json({ status: "received" });
});
app.listen(3000);
```

Then:
```bash
python main.py --source cctv.mp4 --webhook http://localhost:3000/accident-webhook
```

---

## ML Concepts Demonstrated

| Concept | Where Used |
|---|---|
| **Object Detection** | YOLOv8 detects vehicles in each frame |
| **Multi-Object Tracking** | ByteTrack assigns persistent IDs across frames |
| **Computer Vision** | OpenCV for frame I/O, drawing, morphology |
| **Deep Learning** | YOLOv8 backbone (CSPDarknet + PANNet) |
| **Video Processing** | Frame buffering, FPS control, clip assembly |
| **Image Classification** | Aspect-ratio analysis for rollover detection |
| **Signal Processing** | Rolling-average speed analysis for sudden-stop |

---

*Built with Python · OpenCV · YOLOv8 · PyTorch · Streamlit*
