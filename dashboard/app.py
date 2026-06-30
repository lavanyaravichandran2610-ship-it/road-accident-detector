import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import time

st.set_page_config(page_title="Road Accident Detection", page_icon="🚨", layout="wide")
st.title("🚨 Road Accident Detection System")
st.markdown("Real-time accident monitoring powered by YOLOv8 and Computer Vision")
st.markdown("---")

if "accidents" not in st.session_state:
    st.session_state.accidents = 0
if "frames" not in st.session_state:
    st.session_state.frames = 0
if "log" not in st.session_state:
    st.session_state.log = []

uploaded = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
conf = st.sidebar.slider("Confidence", 0.2, 0.9, 0.4)
iou = st.sidebar.slider("Collision IoU", 0.05, 0.5, 0.15)
skip = st.sidebar.slider("Skip frames", 1, 5, 2)

c1, c2, c3 = st.columns(3)
c1.metric("Accidents", st.session_state.accidents)
c2.metric("Frames", st.session_state.frames)
c3.metric("Status", "Ready")

st.markdown("---")
vid_ph = st.empty()
log_ph = st.empty()
st.markdown("---")

b1, b2, b3 = st.columns(3)
start = b1.button("Start Detection", type="primary", use_container_width=True)
stop = b2.button("Stop", use_container_width=True)

if b3.button("Clear Log", use_container_width=True):
    st.session_state.accidents = 0
    st.session_state.frames = 0
    st.session_state.log = []
    st.rerun()

if not st.session_state.log:
    log_ph.info("No accidents detected yet.")
else:
    lines = []
    for e in reversed(st.session_state.log[-10:]):
        lines.append(f"**{e['label']}** {e['confidence']:.0%} at {e['timestamp_s']:.1f}s")
    log_ph.markdown("\n\n".join(lines))

if start:
    if uploaded is None:
        st.error("Please upload a video file first.")
        st.stop()

    import cv2
    import numpy as np

    from modules.video_input import VideoInputHandler
    from modules.vehicle_detection import VehicleDetector
    from modules.accident_detection import AccidentDetector
    from modules.accident_detection import draw_accidents
    from modules.alert_system import AlertManager

    tmp = Path("/tmp") / uploaded.name
    tmp.write_bytes(uploaded.read())

    det = VehicleDetector(model_path="yolov8n.pt", confidence_threshold=conf)
    acc = AccidentDetector(iou_collision_threshold=iou)
    alr = AlertManager(alerts_dir="/tmp/alerts", source_fps=25.0)

    with VideoInputHandler(str(tmp), skip_frames=skip) as vid:
        for frame, meta in vid.stream_frames():
            if stop:
                break
            alr.feed_frame(frame)
            vehicles = det.detect(frame, frame_idx=meta["frame_idx"])
            annotated = det.draw(frame, vehicles)
            events = acc.analyse(
                vehicles,
                frame_idx=meta["frame_idx"],
                timestamp_s=meta["timestamp_s"],
            )
            if events:
                annotated = draw_accidents(annotated, events)
                for ev in events:
                    alr.trigger(ev, annotated)
                    st.session_state.log.append(ev.to_dict())
                    st.session_state.accidents += 1
            st.session_state.frames += 1
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            vid_ph.image(rgb, width=700)

    st.success(f"Done! {st.session_state.accidents} accident(s) detected.")
