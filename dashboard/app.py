import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import time

st.set_page_config(page_title="Traffic Accident Detection System", page_icon="🚦", layout="wide")

st.markdown("""
<div style="background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
            padding:20px;border-radius:12px;border-left:5px solid #e94560;
            margin-bottom:20px;">
  <h1 style="color:white;margin:0;">🚦 Traffic Accident Detection System</h1>
  <p style="color:#a0aec0;margin:5px 0 0;">
    24/7 Live CCTV Monitoring &middot; Auto-Detects Nearest Hospital &amp; Police &middot;
    Prevents Lateral Collisions
  </p>
</div>
""", unsafe_allow_html=True)

if "accidents" not in st.session_state: st.session_state.accidents = 0
if "frames" not in st.session_state: st.session_state.frames = 0
if "log" not in st.session_state: st.session_state.log = []

# ==============================================================================
# CAMERA REGISTRY — Fixed traffic cameras with known RTSP streams
# ==============================================================================

CAMERA_REGISTRY = {
    "CAM-001": {
        "name": "NH47 Main Junction",
        "location": "NH47, Tiruppur, Tamil Nadu",
        "lat": 11.1085,
        "lng": 77.3411,
        "rtsp_url": "",
    },
    "CAM-002": {
        "name": "Avinashi Road Signal",
        "location": "Avinashi Road, Tiruppur",
        "lat": 11.1150,
        "lng": 77.3500,
        "rtsp_url": "",
    },
    "CAM-003": {
        "name": "Palladam Road Junction",
        "location": "Palladam Road, Tiruppur",
        "lat": 11.0950,
        "lng": 77.3300,
        "rtsp_url": "",
    },
}

# ==============================================================================
# SIDEBAR — CAMERA SELECTION (PRIMARY) + TEST MODE (SECONDARY)
# ==============================================================================

with st.sidebar:
    st.markdown("### 🎥 Live Traffic Camera")

    camera_id = st.selectbox(
        "Select Camera Feed",
        list(CAMERA_REGISTRY.keys()),
        format_func=lambda x: f"{x} — {CAMERA_REGISTRY[x]['name']}",
    )
    cam = CAMERA_REGISTRY[camera_id]

    st.markdown(f"""
    **Camera:** {cam['name']}
    **Location:** {cam['location']}
    **GPS:** {cam['lat']}, {cam['lng']}
    """)
    maps_link = f"https://maps.google.com/?q={cam['lat']},{cam['lng']}"
    st.markdown(f"[📍 View camera on Google Maps]({maps_link})")

    st.markdown("---")
    st.markdown("### 📡 Camera Stream Connection")

    rtsp_input = st.text_input(
        f"RTSP URL for {camera_id}",
        value=cam["rtsp_url"],
        placeholder="rtsp://admin:password@192.168.1.100:554/stream",
        help="Enter the live RTSP stream URL of this physical traffic camera",
    )

    video_source = None
    source_label = ""

    if rtsp_input:
        video_source = rtsp_input
        source_label = f"LIVE — {cam['name']} (RTSP)"

    with st.expander("🧪 Test Mode (no live camera connected yet)"):
        st.caption("Use this only for testing/demo when no physical camera is wired up.")
        test_option = st.radio("Test input", ["None", "Upload test video", "Use this device's webcam"], label_visibility="collapsed")

        if test_option == "Upload test video":
            uploaded = st.file_uploader("Upload test footage", type=["mp4", "avi", "mov"])
            if uploaded:
                tmp = Path("/tmp") / uploaded.name
                tmp.write_bytes(uploaded.read())
                video_source = str(tmp)
                source_label = f"TEST — Uploaded file ({uploaded.name})"

        elif test_option == "Use this device's webcam":
            video_source = 0
            source_label = "TEST — Local webcam"

    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")
    conf = st.slider("Detection confidence", 0.2, 0.9, 0.40)
    iou = st.slider("Collision IoU", 0.05, 0.5, 0.15)
    skip = st.slider("Process every N frames", 1, 5, 2)

    st.markdown("---")
    st.markdown("### 🔔 Alert Settings")
    alert_cooldown = st.slider("Alert cooldown (seconds)", 10, 120, 30)
    st.caption("Minimum time between repeated alerts for the same camera")

# ==============================================================================
# STATUS BAR
# ==============================================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("📷 Camera", cam["name"])
c2.metric("📍 Location", cam["location"][:22] + "...")
c3.metric("🚨 Accidents", st.session_state.accidents)
c4.metric("🎞️ Frames", st.session_state.frames)

if video_source is None:
    st.warning(
        f"**{camera_id}** is not connected. Enter the RTSP URL above to connect "
        f"this camera, or use Test Mode for a demo."
    )
else:
    st.success(f"Source ready: {source_label}")

st.markdown("---")

col_video, col_log = st.columns([3, 2])
with col_video:
    st.markdown(f"### 📹 Live Feed — {cam['name']}")
    vid_ph = st.empty()
    prog_ph = st.progress(0.0)
with col_log:
    st.markdown("### 🚨 Accident & Dispatch Log")
    log_ph = st.empty()

st.markdown("---")

b1, b2, b3 = st.columns(3)
start = b1.button("▶ Start Monitoring", type="primary", use_container_width=True)
stop = b2.button("⏹ Stop", use_container_width=True)
if b3.button("🗑 Clear Log", use_container_width=True):
    st.session_state.accidents = 0
    st.session_state.frames = 0
    st.session_state.log = []
    st.rerun()


def show_log():
    if not st.session_state.log:
        log_ph.info("No accidents detected. System monitoring live...")
        return
    html = ""
    for e in reversed(st.session_state.log[-15:]):
        html += f"""
        <div style="background:#1a1f2e;border-left:4px solid #e94560;
                    border-radius:6px;padding:10px;margin-bottom:8px;">
          <b style="color:#e94560;">{e['label']}</b>
          <span style="color:white;"> {e['confidence']:.0%}</span><br>
          <small style="color:#718096;">
            Frame {e['frame_idx']} | {e['timestamp_s']:.1f}s | {e.get('notes', '')}
          </small>
        </div>"""
    log_ph.markdown(html, unsafe_allow_html=True)


show_log()

# ==============================================================================
# MONITORING LOOP
# ==============================================================================

if start:
    if video_source is None:
        st.error(
            f"Cannot start — {camera_id} has no connected stream. "
            f"Enter an RTSP URL or enable Test Mode in the sidebar."
        )
        st.stop()

    import cv2
    import numpy as np

    from modules.video_input import VideoInputHandler
    from modules.vehicle_detection import VehicleDetector
    from modules.accident_detection import AccidentDetector
    from modules.accident_detection import draw_accidents
    from modules.alert_system import AlertManager
    import utils.emergency_alert as ea
    from utils.emergency_alert import trigger_emergency_response

    ea.CAMERA_LOCATION["name"] = cam["location"]
    ea.CAMERA_LOCATION["lat"] = cam["lat"]
    ea.CAMERA_LOCATION["lng"] = cam["lng"]

    det = VehicleDetector(model_path="yolov8n.pt", confidence_threshold=conf)
    acc = AccidentDetector(iou_collision_threshold=iou, cooldown_frames=int(alert_cooldown * 25))
    alr = AlertManager(alerts_dir="/tmp/alerts", source_fps=25.0)

    last_alert_time = 0

    st.info(f"Monitoring started | {source_label} | {cam['location']}")

    with VideoInputHandler(video_source, skip_frames=skip) as vid:
        for frame, meta in vid.stream_frames():
            if stop:
                break

            alr.feed_frame(frame)
            vehicles = det.detect(frame, frame_idx=meta["frame_idx"])
            annotated = det.draw(frame, vehicles)

            cv2.putText(
                annotated,
                f"{cam['name']} | {meta['timestamp_s']:.1f}s",
                (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
            )

            events = acc.analyse(vehicles, frame_idx=meta["frame_idx"], timestamp_s=meta["timestamp_s"])

            if events:
                annotated = draw_accidents(annotated, events)
                for ev in events:
                    screenshot_path = alr.trigger(ev, annotated)
                    st.session_state.log.append(ev.to_dict())
                    st.session_state.accidents += 1

                    now = time.time()
                    if now - last_alert_time > alert_cooldown:
                        last_alert_time = now
                        try:
                            trigger_emergency_response(
                                accident_type=ev.label,
                                confidence=ev.confidence,
                                timestamp_s=ev.timestamp_s,
                                screenshot_path=str(screenshot_path),
                            )
                            st.toast(f"🚨 Hospital & Police notified! {ev.label}", icon="🚨")
                        except Exception as ex:
                            st.warning(f"Alert error: {ex}")

            st.session_state.frames += 1
            prog_ph.progress(min(vid.progress, 1.0))

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            vid_ph.image(rgb, width=700)
            show_log()

    st.success(
        f"Monitoring stopped | {cam['name']} | "
        f"Accidents: {st.session_state.accidents} | Frames: {st.session_state.frames}"
    )
else:
    show_log()
