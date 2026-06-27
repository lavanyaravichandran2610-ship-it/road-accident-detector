"""
Module 1: Video Input
Handles reading CCTV footage, webcam feeds, and video files.
Processes frames one by one and yields them for downstream processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class VideoSource:
    """Enum-like class for video source types."""
    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"   # IP camera / CCTV stream


class VideoInputHandler:
    """
    Reads and preprocesses video from files, webcams, or RTSP streams.

    Usage:
        handler = VideoInputHandler("road_video.mp4")
        for frame, meta in handler.stream_frames():
            # process frame
    """

    def __init__(
        self,
        source,
        target_fps: int = 15,
        resize_to: Optional[Tuple[int, int]] = (1280, 720),
        skip_frames: int = 1,
    ):
        """
        Args:
            source:      Path to video file, int for webcam index, or RTSP URL string.
            target_fps:  Desired processing rate (skips frames to match).
            resize_to:   (width, height) to resize each frame, or None to keep original.
            skip_frames: Process every Nth frame (1 = every frame).
        """
        self.source = source
        self.target_fps = target_fps
        self.resize_to = resize_to
        self.skip_frames = max(1, skip_frames)

        self.cap: Optional[cv2.VideoCapture] = None
        self._source_type = self._detect_source_type(source)

        self.total_frames: int = 0
        self.current_frame_idx: int = 0
        self.source_fps: float = 30.0
        self.source_width: int = 0
        self.source_height: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the video capture device / file. Returns True on success."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error("Could not open video source: %s", self.source)
            return False

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.source_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Opened %s | %.1f fps | %dx%d | %d frames",
            self.source, self.source_fps,
            self.source_width, self.source_height,
            self.total_frames,
        )
        return True

    def close(self):
        """Release the video capture."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def stream_frames(self) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """
        Generator that yields (frame, metadata) tuples.

        Metadata dict contains:
            frame_idx   – absolute frame index in the source
            timestamp_s – position in seconds
            source_fps  – original FPS of the source
            width, height – dimensions *after* resize
        """
        if self.cap is None:
            if not self.open():
                return

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.info("End of stream reached at frame %d.", frame_idx)
                break

            frame_idx += 1
            self.current_frame_idx = frame_idx

            # Skip frames to match desired processing rate
            if frame_idx % self.skip_frames != 0:
                continue

            # Optional resize
            if self.resize_to:
                frame = cv2.resize(frame, self.resize_to)

            timestamp_s = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            meta = {
                "frame_idx": frame_idx,
                "timestamp_s": round(timestamp_s, 3),
                "source_fps": self.source_fps,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "source": str(self.source),
            }

            yield frame, meta

    def get_single_frame(self) -> Optional[Tuple[np.ndarray, dict]]:
        """Read exactly one frame (useful for thumbnail extraction)."""
        for frame, meta in self.stream_frames():
            return frame, meta
        return None

    def seek(self, frame_number: int):
        """Jump to a specific frame index (only works for file sources)."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    @property
    def progress(self) -> float:
        """Returns 0.0–1.0 completion ratio (file sources only)."""
        if self.total_frames > 0:
            return self.current_frame_idx / self.total_frames
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_source_type(source) -> str:
        if isinstance(source, int):
            return VideoSource.WEBCAM
        s = str(source)
        if s.startswith("rtsp://") or s.startswith("rtmp://") or s.startswith("http://"):
            return VideoSource.RTSP
        return VideoSource.FILE

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def list_available_cameras(max_check: int = 5) -> list[int]:
    """Return indices of all connected webcams."""
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def extract_thumbnail(video_path: str, second: float = 1.0) -> Optional[np.ndarray]:
    """Extract a single frame from a video file at `second` seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def overlay_info(frame: np.ndarray, meta: dict) -> np.ndarray:
    """Burn timestamp and frame index onto a frame (non-destructive copy)."""
    out = frame.copy()
    text = f"Frame {meta['frame_idx']} | {meta['timestamp_s']:.2f}s"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out
