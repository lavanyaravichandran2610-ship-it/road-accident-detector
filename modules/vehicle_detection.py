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

# COCO class IDs for vehicle types
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Drawing colours per class (BGR)
CLASS_COLORS = {
    "car":        (0, 200, 255),
    "motorcycle": (0, 165, 255),
    "bus":        (0, 255, 100),
    "truck":      (200, 0, 255),
    "unknown":    (180, 180, 180),
}

HISTORY_LEN = 30   # frames of position history kept per vehicle


@dataclass
class Vehicle:
    """Tracks a single detected vehicle across frames."""

    track_id: int
    class_name: str
    confidence: float

    # Bounding box in current frame [x1, y1, x2, y2]
    bbox: np.ndarray

    # Centre-point history (deque of (cx, cy) tuples)
    positions: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))

    # Estimated pixel-speed history
    speeds: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))

    # Frame index of the last update
    last_seen_frame: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, (x2 - x1) * (y2 - y1))

    @property
    def avg_speed(self) -> float:
        if len(self.speeds) == 0:
            return 0.0
        return float(np.mean(list(self.speeds)))

    @property
    def recent_speed(self) -> float:
        """Speed averaged over the last 5 frames."""
        recent = list(self.speeds)[-5:]
        return float(np.mean(recent)) if recent else 0.0

    def update(self, bbox: np.ndarray, confidence: float, frame_idx: int):
        """Refresh position, speed estimate, and metadata."""
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

    def speed_dropped_suddenly(self, threshold_ratio: float = 0.2) -> bool:
        """
        True if the vehicle's speed in the last 3 frames fell to < `threshold_ratio`
        of its rolling average — indicating a sudden stop after impact.
        """
        if len(self.speeds) < 8:
            return False
        history_avg = float(np.mean(list(self.speeds)[:-3]))
        recent_avg = float(np.mean(list(self.speeds)[-3:]))
        if history_avg < 3:            # vehicle was already slow / stationary
            return False
        return recent_avg < history_avg * threshold_ratio


class VehicleDetector:
    """
    Wraps YOLOv8 to detect & track vehicles.

    Requires `ultralytics` package:
        pip install ultralytics

    Usage:
        detector = VehicleDetector("yolov8n.pt")
        vehicles = detector.detect(frame, frame_idx=42)
        annotated = detector.draw(frame, vehicles)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: str = "cpu",        # "cuda" if GPU available
    ):
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.device = device
        self._model = None
        self._model_path = model_path
        self._vehicle_registry: dict[int, Vehicle] = {}

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info("YOLOv8 model loaded from '%s'.", self._model_path)
        except ImportError:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Vehicle]:
        """
        Run YOLO on `frame` and return a list of tracked Vehicle objects.
        Maintains an internal registry so vehicles carry their history.
        """
       results = self.model.predict(
   	 frame,
   	 conf=self.conf,
  	 iou=self.iou,
   	 device=self.device,
   	 verbose=False,
)

        detected: list[Vehicle] = []

        if not results or results[0].boxes is None:
            return detected

        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

            track_id = idx  # use index as ID since tracking is disabled
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(float)
            class_name = VEHICLE_CLASS_IDS[cls_id]

            if track_id in self._vehicle_registry:
                v = self._vehicle_registry[track_id]
                v.update(bbox, conf, frame_idx)
            else:
                v = Vehicle(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox,
                    last_seen_frame=frame_idx,
                )
                v.positions.append(v.center)
                v.speeds.append(0.0)
                self._vehicle_registry[track_id] = v

            detected.append(self._vehicle_registry[track_id])

        # Prune vehicles not seen for > 60 frames
        stale_ids = [
            tid for tid, v in self._vehicle_registry.items()
            if frame_idx - v.last_seen_frame > 60
        ]
        for tid in stale_ids:
            del self._vehicle_registry[tid]

        return detected

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, vehicles: list[Vehicle]) -> np.ndarray:
        """Return a copy of the frame with bounding boxes and track IDs drawn."""
        out = frame.copy()
        for v in vehicles:
            x1, y1, x2, y2 = map(int, v.bbox)
            color = CLASS_COLORS.get(v.class_name, CLASS_COLORS["unknown"])

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"#{v.track_id} {v.class_name} {v.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Trail
            pts = list(v.positions)
            for i in range(1, len(pts)):
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(out, pt1, pt2, color, 1)

        # Vehicle count HUD
        count_text = f"Vehicles: {len(vehicles)}"
        cv2.putText(out, count_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    # Fallback: simple background-subtraction detector (no GPU / model)
    # ------------------------------------------------------------------

    @staticmethod
    def create_bg_subtractor_detector():
        """
        Returns a lightweight callable that uses MOG2 background subtraction
        to yield pseudo-vehicle bounding boxes when YOLOv8 is unavailable.
        """
        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=True
        )

        def detect_simple(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
            fgmask = fgbg.apply(frame)
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            boxes = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 1500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    boxes.append((x, y, x + w, y + h))
            return boxes

        return detect_simple
