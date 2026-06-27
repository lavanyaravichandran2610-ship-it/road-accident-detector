"""
Module 3: Accident Detection
Analyses detected vehicles frame-by-frame to identify:
  1. Collision     – two or more bounding boxes overlap significantly
  2. Sudden stop   – vehicle speed drops sharply after a period of motion
  3. Roll / flip   – bounding-box aspect ratio changes dramatically
"""

from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
import time

from modules.vehicle_detection import Vehicle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Accident types
# ---------------------------------------------------------------------------

class AccidentType(Enum):
    COLLISION    = auto()   # vehicles overlap (impact)
    SUDDEN_STOP  = auto()   # sharp deceleration post-motion
    ROLLOVER     = auto()   # aspect-ratio flip (landscape → portrait or vice-versa)
    MULTI_VEHICLE = auto()  # 3+ vehicles involved


ACCIDENT_LABELS = {
    AccidentType.COLLISION:     "Vehicle Collision",
    AccidentType.SUDDEN_STOP:   "Sudden Stop / Impact",
    AccidentType.ROLLOVER:      "Vehicle Rollover / Flip",
    AccidentType.MULTI_VEHICLE: "Multi-Vehicle Accident",
}

ACCIDENT_COLORS = {          # BGR for OpenCV overlays
    AccidentType.COLLISION:     (0, 0, 255),
    AccidentType.SUDDEN_STOP:   (0, 128, 255),
    AccidentType.ROLLOVER:      (0, 0, 200),
    AccidentType.MULTI_VEHICLE: (0, 0, 180),
}


# ---------------------------------------------------------------------------
# Accident event data-class
# ---------------------------------------------------------------------------

@dataclass
class AccidentEvent:
    """Describes a single detected accident."""

    accident_type: AccidentType
    frame_idx: int
    timestamp_s: float
    confidence: float               # 0.0 – 1.0
    involved_vehicle_ids: list[int] = field(default_factory=list)
    location_px: Optional[tuple[float, float]] = None  # centre pixel
    notes: str = ""

    @property
    def label(self) -> str:
        return ACCIDENT_LABELS[self.accident_type]

    @property
    def color(self) -> tuple[int, int, int]:
        return ACCIDENT_COLORS[self.accident_type]

    def to_dict(self) -> dict:
        return {
            "type": self.accident_type.name,
            "label": self.label,
            "frame_idx": self.frame_idx,
            "timestamp_s": self.timestamp_s,
            "confidence": round(self.confidence, 3),
            "vehicle_ids": self.involved_vehicle_ids,
            "location_px": self.location_px,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Accident Detector
# ---------------------------------------------------------------------------

class AccidentDetector:
    """
    Stateful detector that consumes Vehicle objects from consecutive frames
    and emits AccidentEvent objects when accident conditions are met.

    Design notes
    ────────────
    • Uses a cooldown mechanism to prevent the same accident from being
      reported on every frame once triggered.
    • All thresholds are configurable at construction time.
    """

    def __init__(
        self,
        iou_collision_threshold: float = 0.15,   # overlap ratio to flag collision
        speed_drop_ratio: float = 0.20,           # speed must drop to <20 % of avg
        rollover_ratio_change: float = 0.55,      # aspect-ratio change to flag rollover
        cooldown_frames: int = 45,                # min frames between same event alerts
        min_speed_for_stop: float = 4.0,          # px/frame – ignore nearly-stationary
        min_vehicle_history: int = 8,             # frames of history required
    ):
        self.iou_threshold = iou_collision_threshold
        self.speed_drop_ratio = speed_drop_ratio
        self.rollover_ratio_change = rollover_ratio_change
        self.cooldown_frames = cooldown_frames
        self.min_speed = min_speed_for_stop
        self.min_history = min_vehicle_history

        # Track last-fired frame per (type, vehicle_ids) to enforce cooldown
        self._last_event: dict[str, int] = {}

        # Vehicle bbox history for rollover detection: {track_id: deque of aspect-ratios}
        from collections import deque
        self._aspect_history: dict[int, deque] = {}
        self._deque_maxlen = 20

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyse(
        self,
        vehicles: list[Vehicle],
        frame_idx: int,
        timestamp_s: float,
    ) -> list[AccidentEvent]:
        """
        Given the vehicles detected in one frame, return any AccidentEvents.
        """
        events: list[AccidentEvent] = []

        # 1. Collision detection
        collision_events = self._check_collisions(vehicles, frame_idx, timestamp_s)
        events.extend(collision_events)

        # 2. Sudden stop detection
        for v in vehicles:
            ev = self._check_sudden_stop(v, frame_idx, timestamp_s)
            if ev:
                events.append(ev)

        # 3. Rollover detection
        for v in vehicles:
            ev = self._check_rollover(v, frame_idx, timestamp_s)
            if ev:
                events.append(ev)

        # 4. Upgrade single collision to MULTI_VEHICLE if 3+ vehicles
        if collision_events:
            involved_ids = set()
            for e in collision_events:
                involved_ids.update(e.involved_vehicle_ids)
            if len(involved_ids) >= 3:
                key = self._event_key(AccidentType.MULTI_VEHICLE, list(involved_ids))
                if self._can_fire(key, frame_idx):
                    cx = np.mean([v.center[0] for v in vehicles if v.track_id in involved_ids])
                    cy = np.mean([v.center[1] for v in vehicles if v.track_id in involved_ids])
                    events.append(AccidentEvent(
                        accident_type=AccidentType.MULTI_VEHICLE,
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        confidence=0.90,
                        involved_vehicle_ids=list(involved_ids),
                        location_px=(float(cx), float(cy)),
                        notes=f"{len(involved_ids)} vehicles involved",
                    ))
                    self._last_event[key] = frame_idx

        return events

    # ------------------------------------------------------------------
    # Detection sub-checks
    # ------------------------------------------------------------------

    def _check_collisions(
        self,
        vehicles: list[Vehicle],
        frame_idx: int,
        timestamp_s: float,
    ) -> list[AccidentEvent]:
        events = []
        n = len(vehicles)
        for i in range(n):
            for j in range(i + 1, n):
                va, vb = vehicles[i], vehicles[j]
                iou = self._compute_iou(va.bbox, vb.bbox)
                if iou < self.iou_threshold:
                    continue

                key = self._event_key(AccidentType.COLLISION, [va.track_id, vb.track_id])
                if not self._can_fire(key, frame_idx):
                    continue

                # Confidence scales with overlap and the speeds involved
                speed_factor = min(1.0, (va.recent_speed + vb.recent_speed) / 30.0)
                confidence = min(0.99, 0.55 + iou * 0.25 + speed_factor * 0.20)

                cx = (va.center[0] + vb.center[0]) / 2
                cy = (va.center[1] + vb.center[1]) / 2

                events.append(AccidentEvent(
                    accident_type=AccidentType.COLLISION,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    confidence=round(confidence, 3),
                    involved_vehicle_ids=[va.track_id, vb.track_id],
                    location_px=(cx, cy),
                    notes=f"IoU={iou:.2f} | speeds {va.recent_speed:.1f}/{vb.recent_speed:.1f} px/fr",
                ))
                self._last_event[key] = frame_idx

        return events

    def _check_sudden_stop(
        self,
        v: Vehicle,
        frame_idx: int,
        timestamp_s: float,
    ) -> Optional[AccidentEvent]:
        if len(v.speeds) < self.min_history:
            return None
        history_avg = float(np.mean(list(v.speeds)[:-3]))
        if history_avg < self.min_speed:
            return None          # was already slow

        recent_avg = float(np.mean(list(v.speeds)[-3:]))
        if recent_avg >= history_avg * self.speed_drop_ratio:
            return None

        key = self._event_key(AccidentType.SUDDEN_STOP, [v.track_id])
        if not self._can_fire(key, frame_idx):
            return None

        ratio = recent_avg / max(history_avg, 1e-6)
        confidence = min(0.92, 0.60 + (1.0 - ratio) * 0.35)

        self._last_event[key] = frame_idx
        return AccidentEvent(
            accident_type=AccidentType.SUDDEN_STOP,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            confidence=round(confidence, 3),
            involved_vehicle_ids=[v.track_id],
            location_px=v.center,
            notes=f"Speed: {history_avg:.1f} → {recent_avg:.1f} px/fr",
        )

    def _check_rollover(
        self,
        v: Vehicle,
        frame_idx: int,
        timestamp_s: float,
    ) -> Optional[AccidentEvent]:
        from collections import deque

        x1, y1, x2, y2 = v.bbox
        w, h = (x2 - x1), (y2 - y1)
        if h == 0:
            return None
        aspect = w / h   # >1 = landscape (normal vehicle), <1 = portrait (flipped)

        if v.track_id not in self._aspect_history:
            self._aspect_history[v.track_id] = deque(maxlen=self._deque_maxlen)
        hist = self._aspect_history[v.track_id]
        hist.append(aspect)

        if len(hist) < self.min_history:
            return None

        baseline = float(np.mean(list(hist)[:-3]))
        current  = float(np.mean(list(hist)[-3:]))

        # A rollover shows as a large change in aspect ratio
        if baseline < 0.5:       # vehicle was already portrait – not a flip
            return None

        ratio_change = abs(current - baseline) / max(baseline, 1e-6)
        if ratio_change < self.rollover_ratio_change:
            return None

        key = self._event_key(AccidentType.ROLLOVER, [v.track_id])
        if not self._can_fire(key, frame_idx):
            return None

        confidence = min(0.95, 0.55 + ratio_change * 0.30)
        self._last_event[key] = frame_idx
        return AccidentEvent(
            accident_type=AccidentType.ROLLOVER,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            confidence=round(confidence, 3),
            involved_vehicle_ids=[v.track_id],
            location_px=v.center,
            notes=f"Aspect ratio {baseline:.2f} → {current:.2f}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        """Intersection-over-Union of two [x1,y1,x2,y2] arrays."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)

    @staticmethod
    def _event_key(atype: AccidentType, ids: list[int]) -> str:
        return f"{atype.name}_{'_'.join(str(i) for i in sorted(ids))}"

    def _can_fire(self, key: str, frame_idx: int) -> bool:
        last = self._last_event.get(key, -999)
        return (frame_idx - last) >= self.cooldown_frames


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_accidents(frame: np.ndarray, events: list[AccidentEvent]) -> np.ndarray:
    """Overlay accident alerts and bounding highlights on a frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    for ev in events:
        color = ev.color

        # Highlight location
        if ev.location_px:
            cx, cy = int(ev.location_px[0]), int(ev.location_px[1])
            cv2.circle(out, (cx, cy), 40, color, 3)
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 30, 2)

        # Banner at top
        banner_h = 55
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
        out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)

        label = f"⚠  ACCIDENT DETECTED: {ev.label}  |  Conf: {ev.confidence:.0%}  |  {ev.timestamp_s:.1f}s"
        cv2.putText(out, label, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return out
