"""
Module 2: Vehicle Detection
Detects and tracks vehicles in each frame using YOLOv8.
Maintains per-vehicle history for speed estimation and movement analysis.
"""

import numpy as np
import cv2
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)

VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLORS = {
    "car": (0, 200, 255),
    "motorcycle": (0, 165, 255),
    "bus": (0, 255, 100),
    "truck": (200, 0, 255),
    "unknown": (180, 180, 180),
}

HISTORY_LEN = 30


@dataclass
class Vehicle:
    track_id: int
    class_name: str
    confidence: float
    bbox: np.ndarray
    positions: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    speeds: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    last_seen_frame: int = 0

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return max(0, (x2 - x1) * (y2 - y1))

    @property
    def avg_speed(self):
        if len(self.speeds) == 0:
            return 0.0
        return float(np.mean(list(self.speeds)))

    @property
    def recent_speed(self):
        recent = list(self.speeds)[-5:]
        return float(np.mean(recent)) if recent else 0.0

    def update(self, bbox, confidence, frame_idx):
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen_frame = frame_idx
        cx, cy = self.center
        if self.positions:
            prev_cx, prev_cy = self.positions[-1]
            pixel_dist = np.hypot(cx - prev_cx, cy - prev_cy)
            self.speeds.append(pixel_dist)
        else:
            self.speeds.append(0.0)
        self.positions.append((cx, cy))

    def speed_dropped_suddenly(self, threshold_ratio=0.2):
        if len(self.speeds) < 8:
            return False
        history_avg = float(np.mean(list(self.speeds)[:-3]))
        recent_avg = float(np.mean(list(self.speeds)[-3:]))
        if history_avg < 3:
            return False
        return recent_avg < history_avg * threshold_ratio


class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.40, iou_threshold=0.45, device="cpu"):
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.device = device
        self._model = None
        self._model_path = model_path
        self._vehicle_registry = {}

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info("YOLOv8 model loaded from '%s'.", self._model_path)
        except ImportError:
            raise RuntimeError("ultralytics is not installed.")

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def detect(self, frame, frame_idx=0):
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)

        detected = []

        if not results or results[0].boxes is None:
            return detected

        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

            track_id = idx
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(float)
            class_name = VEHICLE_CLASS_IDS[cls_id]

            if track_id in self._vehicle_registry:
                v = self._vehicle_registry[track_id]
                v.update(bbox, conf, frame_idx)
            else:
                v = Vehicle(track_id=track_id, class_name=class_name, confidence=conf, bbox=bbox, last_seen_frame=frame_idx)
                v.positions.append(v.center)
                v.speeds.append(0.0)
                self._vehicle_registry[track_id] = v

            detected.append(self._vehicle_registry[track_id])

        stale_ids = [tid for tid, v in self._vehicle_registry.items() if frame_idx - v.last_seen_frame > 60]
        for tid in stale_ids:
            del self._vehicle_registry[tid]

        return detected

    def draw(self, frame, vehicles):
        out = frame.copy()
        for v in vehicles:
            x1, y1, x2, y2 = map(int, v.bbox)
            color = CLASS_COLORS.get(v.class_name, CLASS_COLORS["unknown"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"#{v.track_id} {v.class_name} {v.confidence:.0%}"
            cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(out, f"Vehicles: {len(vehicles)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
        return out
