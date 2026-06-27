"""
Module 4: Alert Generation
Handles all outputs when an accident is detected:
  - Console / log alert
  - Screenshot save (annotated frame)
  - Video clip save (configurable seconds before/after event)
  - JSON event log
  - (Optional) HTTP webhook / SMS / email hook
"""

from __future__ import annotations

import cv2
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from modules.accident_detection import AccidentEvent

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages all alert outputs for accident events.

    Parameters
    ──────────
    alerts_dir      – root directory where all alert artefacts are saved
    clip_pre_s      – seconds of video before the accident to include in clip
    clip_post_s     – seconds of video after the accident to include in clip
    source_fps      – FPS of the incoming video (needed for clip buffering)
    webhook_url     – optional HTTP POST endpoint for real-time notification
    """

    def __init__(
        self,
        alerts_dir: str = "alerts",
        clip_pre_s: float = 5.0,
        clip_post_s: float = 5.0,
        source_fps: float = 25.0,
        webhook_url: Optional[str] = None,
    ):
        self.alerts_dir = Path(alerts_dir)
        self.clip_pre_s = clip_pre_s
        self.clip_post_s = clip_post_s
        self.source_fps = source_fps
        self.webhook_url = webhook_url

        # Pre-event frame buffer for clip creation
        pre_frames = int(clip_pre_s * source_fps) + 10
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=pre_frames)
        self._post_buffer_active = False
        self._post_frames_needed = int(clip_post_s * source_fps)
        self._post_frames_collected: list[np.ndarray] = []
        self._pending_clip_event: Optional[AccidentEvent] = None

        # JSON log
        self._event_log: list[dict] = []
        self._log_path = self.alerts_dir / "event_log.json"

        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_frame(self, frame: np.ndarray):
        """
        Must be called for every frame (even when no accident is detected)
        so the pre-event buffer stays populated.
        """
        self._frame_buffer.append(frame.copy())

        # Collect post-event frames for the pending clip
        if self._post_buffer_active:
            self._post_frames_collected.append(frame.copy())
            if len(self._post_frames_collected) >= self._post_frames_needed:
                self._finalise_clip()

    def trigger(self, event: AccidentEvent, annotated_frame: np.ndarray):
        """
        Called when an accident is confirmed. Fires all configured alert channels.
        Returns the path to the saved screenshot.
        """
        ts = self._timestamp_str()
        event_id = f"{event.accident_type.name}_{ts}"

        logger.warning(
            "🚨 ACCIDENT DETECTED [%s] | frame=%d | t=%.2fs | conf=%.0f%%",
            event.label, event.frame_idx, event.timestamp_s, event.confidence * 100,
        )

        # 1. Save screenshot
        screenshot_path = self._save_screenshot(annotated_frame, event_id, event)

        # 2. Start collecting post-event frames for video clip
        if not self._post_buffer_active:
            self._pending_clip_event = event
            self._post_frames_collected = []
            self._post_buffer_active = True

        # 3. Append to JSON log
        record = {**event.to_dict(), "event_id": event_id, "screenshot": str(screenshot_path)}
        self._event_log.append(record)
        self._save_log()

        # 4. Optional webhook
        if self.webhook_url:
            self._send_webhook(record)

        # 5. Emergency response — SMS + Email + nearest hospital
        try:
            from utils.emergency_alert import trigger_emergency_response
            trigger_emergency_response(
                accident_type=event.label,
                confidence=event.confidence,
                timestamp_s=event.timestamp_s,
                screenshot_path=str(screenshot_path),
            )
        except Exception as exc:
            logger.warning("Emergency alert skipped: %s", exc)

        return screenshot_path

    def get_all_events(self) -> list[dict]:
        """Return all logged accident events."""
        return list(self._event_log)

    def get_screenshot_paths(self) -> list[Path]:
        """Return paths of all saved screenshots."""
        shots_dir = self.alerts_dir / "screenshots"
        return sorted(shots_dir.glob("*.jpg")) if shots_dir.exists() else []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self):
        for sub in ("screenshots", "clips"):
            (self.alerts_dir / sub).mkdir(parents=True, exist_ok=True)

    def _timestamp_str(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

    def _save_screenshot(
        self,
        frame: np.ndarray,
        event_id: str,
        event: AccidentEvent,
    ) -> Path:
        path = self.alerts_dir / "screenshots" / f"{event_id}.jpg"

        # Burn extra info onto the frame
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        info_lines = [
            f"Event ID : {event_id}",
            f"Type     : {event.label}",
            f"Conf     : {event.confidence:.0%}",
            f"Frame    : {event.frame_idx}",
            f"Time     : {event.timestamp_s:.2f}s",
        ]
        y = h - 130
        cv2.rectangle(annotated, (0, y - 10), (320, h), (0, 0, 0), -1)
        for line in info_lines:
            cv2.putText(annotated, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 255, 200), 1, cv2.LINE_AA)
            y += 22

        cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logger.info("Screenshot saved: %s", path)
        return path

    def _finalise_clip(self):
        """Write pre + post frames as an MP4 clip."""
        if self._pending_clip_event is None:
            self._reset_post_buffer()
            return

        event = self._pending_clip_event
        ts = self._timestamp_str()
        clip_path = self.alerts_dir / "clips" / f"{event.accident_type.name}_{ts}.mp4"

        all_frames = list(self._frame_buffer) + self._post_frames_collected
        if all_frames:
            h, w = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(clip_path), fourcc, self.source_fps, (w, h))
            for f in all_frames:
                writer.write(f)
            writer.release()
            logger.info("Accident clip saved: %s (%d frames)", clip_path, len(all_frames))

        self._reset_post_buffer()

    def _reset_post_buffer(self):
        self._post_buffer_active = False
        self._post_frames_collected = []
        self._pending_clip_event = None

    def _save_log(self):
        try:
            with open(self._log_path, "w") as f:
                json.dump(self._event_log, f, indent=2)
        except Exception as exc:
            logger.error("Could not save event log: %s", exc)

    def _send_webhook(self, payload: dict):
        """Fire-and-forget HTTP POST to the configured webhook URL."""
        try:
            import requests
            resp = requests.post(self.webhook_url, json=payload, timeout=5)
            logger.info("Webhook response: %d", resp.status_code)
        except Exception as exc:
            logger.warning("Webhook failed: %s", exc)


# ---------------------------------------------------------------------------
# Console alert helpers
# ---------------------------------------------------------------------------

ALERT_BANNER = """
╔══════════════════════════════════════════════════════╗
║          🚨  ROAD ACCIDENT DETECTED  🚨              ║
╚══════════════════════════════════════════════════════╝
"""


def print_alert(event: AccidentEvent):
    """Print a styled console alert for an accident event."""
    print(ALERT_BANNER)
    print(f"  Type       : {event.label}")
    print(f"  Confidence : {event.confidence:.0%}")
    print(f"  Frame      : {event.frame_idx}")
    print(f"  Timestamp  : {event.timestamp_s:.2f} s")
    if event.location_px:
        print(f"  Location   : pixel ({event.location_px[0]:.0f}, {event.location_px[1]:.0f})")
    if event.notes:
        print(f"  Notes      : {event.notes}")
    print()