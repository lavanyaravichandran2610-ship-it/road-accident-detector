"""
Module 5: Streamlit Dashboard
Real-time accident detection interface with:
  - Video upload / webcam / RTSP input
  - Live annotated video feed
  - Accident event log
  - Confidence metrics
  - Screenshot gallery
"""

import json
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Lazy imports — only load heavy libraries when user clicks Start
# This prevents memory crash on startup
import importlib

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Road Accident Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Dark header */
  .main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #e94560;
  }
  .main-header h1 { color: #fff; margin: 0; font-size: 1.9rem; }
  .main-header p  { color: #a0aec0; margin: 0.3rem 0 0; font-size: 0.95rem; }

  /* Metric cards */
  .metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    border: 1px solid #2d3748;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
  .metric-label { font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }

  /* Alert badge */
  .alert-badge {
    display: inline-block;
    background: #e94560;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }

  /* Event row */
  .event-row {
    background: #1a1f2e;
    border-left: 4px solid #e94560;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🚨 Road Accident Detection System</h1>
  <p>Real-time accident monitoring powered by YOLOv8 · Computer Vision · Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

if "total_accidents" not in st.session_state:
    st.session_state.total_accidents = 0
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0
if "event_log" not in st.session_state:
    st.session_state.event_log = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – configuration
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("**Video Source**")
    source_type = st.radio(
        "Source type",
        ["Upload Video File", "Webcam", "RTSP Stream URL"],
        label_visibility="collapsed",
    )

    video_source = None
    if source_type == "Upload Video File":
        uploaded = st.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded:
            # Save to temp file
            tmp_path = Path("/tmp") / uploaded.name
            tmp_path.write_bytes(uploaded.read())
            video_source = str(tmp_path)
            st.success(f"Loaded: {uploaded.name}")

    elif source_type == "Webcam":
        cam_idx = st.number_input("Camera index", min_value=0, max_value=10, value=0)
        video_source = int(cam_idx)

    else:
        rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://192.168.1.100:554/stream")
        if rtsp_url:
            video_source = rtsp_url

    st.markdown("---")
    st.markdown("**Detection Settings**")

    yolo_model = st.selectbox(
        "YOLOv8 model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="n=nano (fast), s=small, m=medium (accurate)",
    )

    conf_threshold = st.slider("Detection confidence", 0.20, 0.90, 0.40, 0.05)
    iou_collision  = st.slider("Collision IoU threshold", 0.05, 0.50, 0.15, 0.05)
    speed_drop     = st.slider("Speed-drop sensitivity", 0.05, 0.50, 0.20, 0.05)

    st.markdown("---")
    st.markdown("**Alerts**")
    save_screenshots = st.checkbox("Save screenshots", value=True)
    save_clips       = st.checkbox("Save video clips", value=True)
    alerts_dir       = st.text_input("Alerts directory", value="alerts")

    webhook_url = st.text_input(
        "Webhook URL (optional)", placeholder="https://your-server/webhook"
    )

    st.markdown("---")
    st.markdown("**Processing**")
    skip_frames = st.slider("Process every N frames", 1, 5, 2)
    max_runtime = st.number_input(
        "Max runtime (seconds, 0=unlimited)", min_value=0, value=0
    )

# ──────────────────────────────────────────────────────────────────────────────
# Metrics row
# ──────────────────────────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)

with m1:
    total_acc_placeholder = st.empty()
with m2:
    total_frames_placeholder = st.empty()
with m3:
    fps_placeholder = st.empty()
with m4:
    status_placeholder = st.empty()


def render_metrics(accidents, frames, fps, status):
    total_acc_placeholder.metric("🚨 Accidents Detected", accidents)
    total_frames_placeholder.metric("🎞️ Frames Processed", frames)
    fps_placeholder.metric("⚡ Processing FPS", f"{fps:.1f}")
    status_placeholder.metric("🟢 Status", status)


render_metrics(
    st.session_state.total_accidents,
    st.session_state.total_frames,
    0.0,
    "Ready",
)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────────────────

col_video, col_log = st.columns([3, 2])

with col_video:
    st.markdown("### 📹 Live Feed")
    video_placeholder = st.empty()
    progress_bar = st.progress(0.0)

with col_log:
    st.markdown("### 📋 Accident Log")
    log_placeholder = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# Screenshot gallery (below main layout)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
gallery_header = st.empty()
gallery_cols_placeholder = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# Control buttons
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 4])

with btn_col1:
    start_btn = st.button("▶  Start Detection", use_container_width=True, type="primary")
with btn_col2:
    stop_btn  = st.button("⏹  Stop", use_container_width=True)
with btn_col3:
    if st.button("🗑  Clear Log", use_container_width=True):
        st.session_state.event_log = []
        st.session_state.total_accidents = 0
        st.session_state.total_frames = 0
        st.rerun()


def _render_log(events: list[dict]):
    if not events:
        log_placeholder.info("No accidents detected yet.")
        return
    html_rows = ""
    for ev in reversed(events[-20:]):
        html_rows += f"""
        <div class="event-row">
          <span class="alert-badge">{ev['label']}</span>
          &nbsp; Conf: <b>{ev['confidence']:.0%}</b>
          &nbsp; | Frame: {ev['frame_idx']}
          &nbsp; | {ev['timestamp_s']:.1f}s
          <br><small style="color:#718096">{ev.get('notes','')}</small>
        </div>"""
    log_placeholder.markdown(html_rows, unsafe_allow_html=True)


def _render_gallery():
    alert_mgr_dir = Path(alerts_dir) / "screenshots"
    shots = sorted(alert_mgr_dir.glob("*.jpg"))[-9:] if alert_mgr_dir.exists() else []
    if not shots:
        return
    gallery_header.markdown("### 🖼️ Accident Screenshots")
    cols = st.columns(min(3, len(shots)))
    for i, path in enumerate(shots):
        with cols[i % 3]:
            img = cv2.imread(str(path))
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption=path.stem[:30], use_column_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────────────────────────────────────

if start_btn:
    # Import heavy modules only when needed
    from modules.video_input import VideoInputHandler, extract_thumbnail
    from modules.vehicle_detection import VehicleDetector
    from modules.accident_detection import AccidentDetector, draw_accidents
    from modules.alert_system import AlertManager, print_alert
    if video_source is None:
        st.error("Please select a video source first.")
        st.stop()

    st.session_state.processing = True

    # Initialise modules
    detector = VehicleDetector(
        model_path=yolo_model,
        confidence_threshold=conf_threshold,
    )
    accident_det = AccidentDetector(
        iou_collision_threshold=iou_collision,
        speed_drop_ratio=speed_drop,
    )
    alert_mgr = AlertManager(
        alerts_dir=alerts_dir,
        source_fps=25.0,
        webhook_url=webhook_url or None,
    )

    start_time = time.time()
    last_fps_time = start_time
    frames_since_fps = 0

    with VideoInputHandler(video_source, skip_frames=skip_frames) as vid:
        for frame, meta in vid.stream_frames():
            # Stop button check
            if stop_btn:
                break

            if max_runtime > 0 and (time.time() - start_time) > max_runtime:
                break

            alert_mgr.feed_frame(frame)

            # Detect vehicles
            vehicles = detector.detect(frame, frame_idx=meta["frame_idx"])

            # Annotate frame with vehicle boxes
            annotated = detector.draw(frame, vehicles)

            # Check for accidents
            events = accident_det.analyse(
                vehicles,
                frame_idx=meta["frame_idx"],
                timestamp_s=meta["timestamp_s"],
            )

            # Overlay accident alerts
            if events:
                annotated = draw_accidents(annotated, events)
                for ev in events:
                    if save_screenshots:
                        alert_mgr.trigger(ev, annotated)
                    rec = ev.to_dict()
                    st.session_state.event_log.append(rec)
                    st.session_state.total_accidents += 1

            # Display frame
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_column_width=True)

            # Update progress
            progress_bar.progress(min(vid.progress, 1.0))

            # Update metrics
            st.session_state.total_frames += 1
            frames_since_fps += 1
            now = time.time()
            elapsed = now - last_fps_time
            fps = frames_since_fps / elapsed if elapsed > 0 else 0
            if elapsed >= 1.0:
                last_fps_time = now
                frames_since_fps = 0

            render_metrics(
                st.session_state.total_accidents,
                st.session_state.total_frames,
                fps,
                "🔴 Running",
            )

            _render_log(st.session_state.event_log)

    _render_gallery()
    render_metrics(
        st.session_state.total_accidents,
        st.session_state.total_frames,
        0.0,
        "✅ Done",
    )
    st.success(
        f"Processing complete. Detected {st.session_state.total_accidents} accident(s) "
        f"in {st.session_state.total_frames} frames."
    )
    st.session_state.processing = False

else:
    _render_log(st.session_state.event_log)
    _render_gallery()
