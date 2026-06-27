"""
Unit Tests — Road Accident Detection System
Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
# Module 1 – VideoInputHandler
# ──────────────────────────────────────────────────────────────────────────────

class TestVideoInputHandler:
    def test_source_type_detection(self):
        from modules.video_input import VideoInputHandler, VideoSource
        assert VideoInputHandler._detect_source_type(0) == VideoSource.WEBCAM
        assert VideoInputHandler._detect_source_type("video.mp4") == VideoSource.FILE
        assert VideoInputHandler._detect_source_type("rtsp://192.168.1.1/stream") == VideoSource.RTSP

    def test_invalid_source_does_not_crash(self):
        from modules.video_input import VideoInputHandler
        handler = VideoInputHandler("nonexistent_file.mp4")
        result = handler.open()
        assert result is False

    def test_progress_zero_when_no_frames(self):
        from modules.video_input import VideoInputHandler
        handler = VideoInputHandler("dummy.mp4")
        assert handler.progress == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Module 2 – Vehicle dataclass
# ──────────────────────────────────────────────────────────────────────────────

class TestVehicle:
    def _make_vehicle(self, bbox=None) -> "Vehicle":
        from modules.vehicle_detection import Vehicle
        v = Vehicle(
            track_id=1,
            class_name="car",
            confidence=0.9,
            bbox=np.array(bbox or [100, 100, 200, 200], dtype=float),
        )
        return v

    def test_center_calculation(self):
        v = self._make_vehicle([100, 100, 200, 200])
        cx, cy = v.center
        assert cx == pytest.approx(150.0)
        assert cy == pytest.approx(150.0)

    def test_area_calculation(self):
        v = self._make_vehicle([0, 0, 100, 50])
        assert v.area == pytest.approx(5000.0)

    def test_speed_history_populated_on_update(self):
        v = self._make_vehicle([0, 0, 100, 100])
        v.positions.append((50, 50))
        v.update(np.array([110, 110, 210, 210], dtype=float), 0.9, frame_idx=1)
        assert len(v.speeds) > 0

    def test_speed_drop_detection(self):
        from modules.vehicle_detection import Vehicle
        v = Vehicle(track_id=99, class_name="car", confidence=0.8,
                    bbox=np.array([0, 0, 100, 100], dtype=float))
        # Simulate fast movement followed by sudden stop
        for i in range(10):
            v.speeds.append(20.0)
        v.speeds.append(0.5)
        v.speeds.append(0.5)
        v.speeds.append(0.5)
        assert v.speed_dropped_suddenly(threshold_ratio=0.2) is True

    def test_no_speed_drop_when_slow(self):
        from modules.vehicle_detection import Vehicle
        v = Vehicle(track_id=7, class_name="truck", confidence=0.7,
                    bbox=np.array([0, 0, 50, 50], dtype=float))
        for _ in range(12):
            v.speeds.append(1.0)   # already slow
        assert v.speed_dropped_suddenly() is False


# ──────────────────────────────────────────────────────────────────────────────
# Module 3 – AccidentDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestAccidentDetector:
    def _make_vehicle(self, tid, bbox, speeds=None):
        from modules.vehicle_detection import Vehicle
        v = Vehicle(track_id=tid, class_name="car", confidence=0.85,
                    bbox=np.array(bbox, dtype=float))
        if speeds:
            for s in speeds:
                v.speeds.append(s)
                v.positions.append(v.center)
        return v

    def test_collision_detected_on_overlap(self):
        from modules.accident_detection import AccidentDetector, AccidentType
        det = AccidentDetector(iou_collision_threshold=0.10, cooldown_frames=0)
        va = self._make_vehicle(1, [100, 100, 300, 300])
        vb = self._make_vehicle(2, [150, 150, 350, 350])   # large overlap
        events = det.analyse([va, vb], frame_idx=10, timestamp_s=1.0)
        types = [e.accident_type for e in events]
        assert AccidentType.COLLISION in types

    def test_no_collision_when_separated(self):
        from modules.accident_detection import AccidentDetector, AccidentType
        det = AccidentDetector(iou_collision_threshold=0.10)
        va = self._make_vehicle(1, [0, 0, 100, 100])
        vb = self._make_vehicle(2, [500, 500, 600, 600])   # no overlap
        events = det.analyse([va, vb], frame_idx=10, timestamp_s=1.0)
        collision_events = [e for e in events if e.accident_type == AccidentType.COLLISION]
        assert collision_events == []

    def test_sudden_stop_detected(self):
        from modules.accident_detection import AccidentDetector, AccidentType
        det = AccidentDetector(speed_drop_ratio=0.20, cooldown_frames=0, min_speed_for_stop=3.0)
        v = self._make_vehicle(3, [200, 200, 300, 300],
                                speeds=[20]*10 + [0.5, 0.5, 0.5])
        events = det.analyse([v], frame_idx=20, timestamp_s=2.0)
        types = [e.accident_type for e in events]
        assert AccidentType.SUDDEN_STOP in types

    def test_iou_computation(self):
        from modules.accident_detection import AccidentDetector
        det = AccidentDetector()
        # Identical boxes → IoU = 1.0
        iou = det._compute_iou(
            np.array([0, 0, 100, 100]),
            np.array([0, 0, 100, 100]),
        )
        assert iou == pytest.approx(1.0, abs=1e-5)

        # Non-overlapping → IoU = 0.0
        iou2 = det._compute_iou(
            np.array([0, 0, 10, 10]),
            np.array([50, 50, 60, 60]),
        )
        assert iou2 == pytest.approx(0.0, abs=1e-5)

    def test_cooldown_prevents_duplicate_alerts(self):
        from modules.accident_detection import AccidentDetector, AccidentType
        det = AccidentDetector(iou_collision_threshold=0.10, cooldown_frames=30)
        va = self._make_vehicle(1, [100, 100, 300, 300])
        vb = self._make_vehicle(2, [150, 150, 350, 350])

        events1 = det.analyse([va, vb], frame_idx=10, timestamp_s=1.0)
        events2 = det.analyse([va, vb], frame_idx=12, timestamp_s=1.1)  # within cooldown

        assert len(events1) >= 1
        # Second call within cooldown should NOT produce a collision event
        collision2 = [e for e in events2 if e.accident_type == AccidentType.COLLISION]
        assert collision2 == []


# ──────────────────────────────────────────────────────────────────────────────
# Module 4 – AlertManager
# ──────────────────────────────────────────────────────────────────────────────

class TestAlertManager:
    def test_event_log_is_empty_initially(self, tmp_path):
        from modules.alert_system import AlertManager
        mgr = AlertManager(alerts_dir=str(tmp_path))
        assert mgr.get_all_events() == []

    def test_screenshot_saved_on_trigger(self, tmp_path):
        from modules.alert_system import AlertManager
        from modules.accident_detection import AccidentEvent, AccidentType
        mgr = AlertManager(alerts_dir=str(tmp_path))

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ev = AccidentEvent(
            accident_type=AccidentType.COLLISION,
            frame_idx=100,
            timestamp_s=4.0,
            confidence=0.88,
        )
        path = mgr.trigger(ev, dummy_frame)
        assert path.exists()

    def test_event_appended_to_log(self, tmp_path):
        from modules.alert_system import AlertManager
        from modules.accident_detection import AccidentEvent, AccidentType
        mgr = AlertManager(alerts_dir=str(tmp_path))

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ev = AccidentEvent(
            accident_type=AccidentType.SUDDEN_STOP,
            frame_idx=50,
            timestamp_s=2.0,
            confidence=0.75,
        )
        mgr.trigger(ev, dummy_frame)
        log = mgr.get_all_events()
        assert len(log) == 1
        assert log[0]["type"] == "SUDDEN_STOP"

    def test_json_log_written_to_disk(self, tmp_path):
        from modules.alert_system import AlertManager
        from modules.accident_detection import AccidentEvent, AccidentType
        import json as _json
        mgr = AlertManager(alerts_dir=str(tmp_path))

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ev = AccidentEvent(
            accident_type=AccidentType.ROLLOVER,
            frame_idx=200,
            timestamp_s=8.0,
            confidence=0.91,
        )
        mgr.trigger(ev, dummy_frame)

        log_path = tmp_path / "event_log.json"
        assert log_path.exists()
        data = _json.loads(log_path.read_text())
        assert data[0]["type"] == "ROLLOVER"


# ──────────────────────────────────────────────────────────────────────────────
# AccidentEvent
# ──────────────────────────────────────────────────────────────────────────────

class TestAccidentEvent:
    def test_to_dict_contains_required_keys(self):
        from modules.accident_detection import AccidentEvent, AccidentType
        ev = AccidentEvent(
            accident_type=AccidentType.COLLISION,
            frame_idx=10,
            timestamp_s=1.0,
            confidence=0.87,
        )
        d = ev.to_dict()
        for key in ("type", "label", "frame_idx", "timestamp_s", "confidence"):
            assert key in d

    def test_label_is_human_readable(self):
        from modules.accident_detection import AccidentEvent, AccidentType
        ev = AccidentEvent(AccidentType.ROLLOVER, 0, 0.0, 0.8)
        assert "Rollover" in ev.label or "Roll" in ev.label or "Flip" in ev.label
