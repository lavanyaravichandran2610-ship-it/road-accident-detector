"""
Module 3: Accident Detection
Analyses detected vehicles frame-by-frame to identify:
  1. Collision     – two or more bounding boxes overlap significantly
  2. Sudden stop   – vehicle speed drops sharply after a period of motion
  3. Roll / flip   – bounding-box aspect ratio changes dramatically
"""

from __future__ import annotations

import numpy as np
import cv2
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)


class AccidentType(Enum):
    COLLISION = auto()
    SUDDEN_STOP = auto()
    ROLLOVER = auto()
    MULTI_VEHICLE = auto()


ACCIDENT_LABELS = {
    AccidentType.COLLISION: "Vehicle Collision",
    AccidentType.SUDDEN_STOP: "Sudden Stop / Impact",
    AccidentType.ROLLOVER: "Vehicle Rollover / Flip",
    AccidentType.MULTI_VEHICLE: "Multi-Vehicle Accident",
}

ACCIDENT_COLORS = {
    AccidentType.COLLISION: (0, 0, 255),
    AccidentType.SUDDEN_STOP: (0, 128, 255),
    AccidentType.ROLLOVER: (0, 0, 200),
    AccidentType.MULTI_VEHICLE: (0, 0, 180),
}


@dataclass
class AccidentEvent:
    accident_type: AccidentType
    frame_idx: int
    timestamp_s: float
    confidence: float
    involved_vehicle_ids: list = field(default_factory=list)
    location_px: Optional[tuple] = None
    notes: str = ""

    @property
    def label(self):
        return ACCIDENT_LABELS[self.accident_type]

    @property
    def color(self):
        return ACCIDENT_COLORS[self.accident_type]

    def to_dict(self):
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


class AccidentDetector:
    def __init__(self, iou_collision_threshold=0.15, speed_drop_ratio=0.20, rollover_ratio_change=0.55,
                 cooldown_frames=45, min_speed_for_stop=4.0, min_vehicle_history=8):
        self.iou_threshold = iou_collision_threshold
        self.speed_drop_ratio = speed_drop_ratio
        self.rollover_ratio_change = rollover_ratio_change
        self.cooldown_frames = cooldown_frames
        self.min_speed = min_speed_for_stop
        self.min_history = min_vehicle_history
        self._last_event = {}
        self._aspect_history = {}
        self._deque_maxlen = 20

    def analyse(self, vehicles, frame_idx, timestamp_s):
        events = []

        collision_events = self._check_collisions(vehicles, frame_idx, timestamp_s)
        events.extend(collision_events)

        for v in vehicles:
            ev = self._check_sudden_stop(v, frame_idx, timestamp_s)
            if ev:
                events.append(ev)

        for v in vehicles:
            ev = self._check_rollover(v, frame_idx, timestamp_s)
            if ev:
                events.append(ev)

        return events

    def _check_collisions(self, vehicles, frame_idx, timestamp_s):
        events = []
        n = len(vehicles)
        for i in range(n):
            for j in range(i + 1, n):
                va = vehicles[i]
                vb = vehicles[j]
                iou = self._compute_iou(va.bbox, vb.bbox)
                if iou < self.iou_threshold:
                    continue
                key = self._event_key(AccidentType.COLLISION, [va.track_id, vb.track_id])
                if not self._can_fire(key, frame_idx):
                    continue
                speed_factor = min(1.0, (va.recent_speed + vb.recent_speed) / 30.0)
                confidence = min(0.99, 0.55 + iou * 0.25 + speed_factor * 0.20)
                cx = float((va.center[0] + vb.center[0]) / 2)
                cy = float((va.center[1] + vb.center[1]) / 2)
                events.append(AccidentEvent(
                    accident_type=AccidentType.COLLISION,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    confidence=round(confidence, 3),
                    involved_vehicle_ids=[va.track_id, vb.track_id],
                    location_px=(cx, cy),
                    notes=f"IoU={iou:.2f}",
                ))
                self._last_event[key] = frame_idx
        return events

    def _check_sudden_stop(self, v, frame_idx, timestamp_s):
        if len(v.speeds) < self.min_history:
            return None
        history_avg = float(np.mean(list(v.speeds)[:-3]))
        if history_avg < self.min_speed:
            return None
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
            location_px=(float(v.center[0]), float(v.center[1])),
            notes=f"Speed: {history_avg:.1f} to {recent_avg:.1f} px/fr",
        )

    def _check_rollover(self, v, frame_idx, timestamp_s):
        from collections import deque
        x1, y1, x2, y2 = v.bbox
        w = float(x2 - x1)
        h = float(y2 - y1)
        if h == 0:
            return None
        aspect = w / h
        if v.track_id not in self._aspect_history:
            self._aspect_history[v.track_id] = deque(maxlen=self._deque_maxlen)
        hist = self._aspect_history[v.track_id]
        hist.append(aspect)
        if len(hist) < self.min_history:
            return None
        baseline = float(np.mean(list(hist)[:-3]))
        current = float(np.mean(list(hist)[-3:]))
        if baseline < 0.5:
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
            location_px=(float(v.center[0]), float(v.center[1])),
            notes=f"Aspect ratio {baseline:.2f} to {current:.2f}",
        )

    @staticmethod
    def _compute_iou(bbox_a, bbox_b):
        ax1, ay1, ax2, ay2 = float(bbox_a[0]), float(bbox_a[1]), float(bbox_a[2]), float(bbox_a[3])
        bx1, by1, bx2, by2 = float(bbox_b[0]), float(bbox_b[1]), float(bbox_b[2]), float(bbox_b[3])
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)

    @staticmethod
    def _event_key(atype, ids):
        return f"{atype.name}_{'_'.join(str(i) for i in sorted(ids))}"

    def _can_fire(self, key, frame_idx):
        last = self._last_event.get(key, -999)
        return (frame_idx - last) >= self.cooldown_frames


def draw_accidents(frame, events):
    out = frame.copy()
    h, w = out.shape[:2]

    for ev in events:
        color = ev.color

        if ev.location_px:
            cx = int(ev.location_px[0])
            cy = int(ev.location_px[1])
            cv2.circle(out, (cx, cy), 40, color, 3)
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 30, 2)

        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), color, -1)
        out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)

        label = f"ACCIDENT: {ev.label} | {ev.confidence:.0%} | {ev.timestamp_s:.1f}s"
        cv2.putText(out, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return out
