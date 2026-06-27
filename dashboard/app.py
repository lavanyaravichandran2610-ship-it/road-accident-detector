import os
import sys
from pathlib import Path

os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st
import json
import time

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Road Accident Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #e94560;
}
.main-header h1 { color: #fff; margin: 0; font-size: 1.9rem; }
.main-header p  { color: #a0aec0; margin: 0.3rem 0 0; font-size: 0.95rem; }
.event-row {
    background: #1a1f2e;
    border-left: 4px solid #e94560;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}
.alert-badge {
    display: inline-block;
    background: #e94560;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
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

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("**Video Source**")

    source_type = st.radio(
        "Source type",
        ["Upload Video File", "Webcam"],
        label_visibility="collapsed",
    )

    video_source = None
    if source_type == "Upload Video File":
        uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            tmp_path = Path("/tmp") / uploaded.name
            tmp_path.write_bytes(uploaded.read())
            video_source = str(tmp_path)
            st.success(f"Loaded: {uploaded.name}")
    else:
        cam_idx = st.number_input("Camera index", min_value=0, max_value=10, value=0)
        video_source = int(cam_idx)

    st.markdown("---")
    st.markdown("**Detection Settings**")
    conf_threshold = st.slider("Detection confidence", 0.20, 0.90, 0.40, 0.05)
    iou_collision  = st.slider("Collision IoU threshold", 0.05, 0.50, 0.15, 0.05)
    skip_frames    = st.slider("Process every N frames", 1, 5, 2)
    alerts_dir     = st.text_input("Alerts directory", value="/tmp/alerts")

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)
total_acc_ph   = m1.empty()
total_frame_ph = m2.empty()
fps_ph         = m3.empty()
status_ph      = m4.empty()

def render_metrics(accidents, frames, fps, status):
    total_acc_ph.metric("🚨 Accidents", accidents)
    total_frame_ph.metric("🎞️ Frames", frames)
    fps_ph.metric("⚡ FPS", f"{fps:.1f}")
    status_ph.metric("Status", status)

render_metrics(st.session_state.total_accidents, st.session_state.total_frames, 0.0, "Ready")

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

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Buttons
# ──────────────────────────────────────────────────────────────────────────────

btn1, btn2, btn3 = st.columns([2, 2, 4])
start_btn = btn1.button("▶  Start Detection", use_container_width=True, type="primary")
stop_btn  = btn2.button("⏹  Stop", use_container_width=True)

if btn3.button("🗑  Clear Log", use_container_width=True):
    st.session_state.event_log = []
    st.session_state.total_accidents = 0
    st.session_state.total_frames = 0
    st.rerun()

def render_log(events):
    if not events:
        log_placeholder.info("No accidents detected yet.")
        return
    html = ""
    for ev in reversed(events[-20:]):
        html += f"""
        <div class="event-row">
          <span class="alert-badge">{ev['label']}</span>
          &nbsp; Conf: <b>{ev['confidence']:.0%}</b>
          &nbsp;| Frame: {ev['frame_idx']}
          &nbsp;| {ev['timestamp_s']:.1f}s
          <br><small style="color:#718096">{ev.get('notes','')}</small>
        </div>"""
    log_placeholder.markdown(html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Processing loop
# ──────────────────────────────────────────────────────────────────────────────

if start_btn:
    if video_source is None:
        st.error("Please select a video source first.")
        st.stop()

    from modules.video_input import VideoInputHandler
    from modules.vehicle_detection import VehicleDetector
    from modules.accident_detection import AccidentDetector, draw_accidents
    from modules.alert_system import AlertManager

    detector     = VehicleDetector(model_path="yolov8n.pt", confidence_threshold=conf_threshold)
    accident_det = AccidentDetector(iou_collision_threshold=iou_collision)
    alert_mgr    = AlertManager(alerts_dir=alerts_dir, source_fps=25.0)

    start_time     = time.time()
    last_fps_time  = start_time
    frames_since   = 0

    with VideoInputHandler(video_source, skip_frames=skip_frames) as vid:
        for frame, meta in vid.stream_frames():
            if stop_btn:
                break

            alert_mgr.feed_frame(frame)

            vehicles = detector.detect(frame, frame_idx=meta["frame_idx"])
            annotated = detector.draw(frame, vehicles)

            events = accident_det.analyse(
                vehicles,
                frame_idx=meta["frame_idx"],
                timestamp_s=meta["timestamp_s"],
            )

            if events:
                annotated = draw_accidents(annotated, events)
                for ev in events:
                    alert_mgr.trigger(ev, annotated)
                    st.session_state.event_log.append(ev.to_dict())
                    st.session_state.total_accidents += 1

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_column_width=True)
            progress_bar.progress(min(vid.progress, 1.0))

            st.session_state.total_frames += 1
            frames_since += 1
            now = time.time()
            elapsed = now - last_fps_time
            fps = frames_since / elapsed if elapsed > 0 else 0
            if elapsed >= 1.0:
                last_fps_time = now
                frames_since = 0

            render_metrics(
                st.session_state.total_accidents,
                st.session_state.total_frames,
                fps,
                "🔴 Running",
            )
            render_log(st.session_state.event_log)

    render_metrics(
        st.session_state.total_accidents,
        st.session_state.total_frames,
        0.0,
        "✅ Done",
    )
    st.success(f"Done! Detected {st.session_state.total_accidents} accident(s) in {st.session_state.total_frames} frames.")

else:
    render_log(st.session_state.event_log)