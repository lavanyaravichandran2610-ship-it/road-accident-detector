"""
Road Accident Detection System — CLI Entry Point
─────────────────────────────────────────────────
Run from the project root:

    # Process a video file
    python main.py --source road_video.mp4

    # Use webcam
    python main.py --source 0

    # RTSP stream
    python main.py --source rtsp://192.168.1.10:554/stream

    # Launch Streamlit dashboard instead
    python main.py --dashboard
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Project imports ───────────────────────────────────────────────────────────
from modules.video_input import VideoInputHandler
from modules.vehicle_detection import VehicleDetector
from modules.accident_detection import AccidentDetector, draw_accidents
from modules.alert_system import AlertManager, print_alert


# ──────────────────────────────────────────────────────────────────────────────
# CLI Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args):
    logger.info("=== Road Accident Detection System ===")
    logger.info("Source      : %s", args.source)
    logger.info("Model       : %s", args.model)
    logger.info("Alerts dir  : %s", args.alerts_dir)

    # ── Initialise modules ───────────────────────────────────────────────────
    vehicle_detector = VehicleDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )
    accident_detector = AccidentDetector(
        iou_collision_threshold=args.iou,
        speed_drop_ratio=args.speed_drop,
        cooldown_frames=args.cooldown,
    )
    alert_manager = AlertManager(
        alerts_dir=args.alerts_dir,
        clip_pre_s=args.clip_pre,
        clip_post_s=args.clip_post,
        webhook_url=args.webhook or None,
    )

    # ── Video source ─────────────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source

    # ── Display window ───────────────────────────────────────────────────────
    show = args.show and not args.headless
    if show:
        cv2.namedWindow("Road Accident Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Road Accident Detection", 1280, 720)

    # ── Optional video writer ─────────────────────────────────────────────────
    output_writer = None
    if args.output:
        out_path = Path(args.output)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # We'll set resolution on first frame

    # ── Main loop ─────────────────────────────────────────────────────────────
    total_frames = 0
    total_accidents = 0
    start_time = time.time()

    try:
        with VideoInputHandler(source, skip_frames=args.skip) as vid:
            for frame, meta in vid.stream_frames():
                alert_manager.feed_frame(frame)

                # ── Vehicle detection ──────────────────────────────────────
                vehicles = vehicle_detector.detect(frame, frame_idx=meta["frame_idx"])

                # ── Draw vehicle boxes ─────────────────────────────────────
                annotated = vehicle_detector.draw(frame, vehicles)

                # ── Accident detection ─────────────────────────────────────
                events = accident_detector.analyse(
                    vehicles,
                    frame_idx=meta["frame_idx"],
                    timestamp_s=meta["timestamp_s"],
                )

                if events:
                    annotated = draw_accidents(annotated, events)
                    for ev in events:
                        print_alert(ev)
                        alert_manager.trigger(ev, annotated)
                        total_accidents += 1

                # ── Frame counter overlay ──────────────────────────────────
                fps_text = f"Frame {meta['frame_idx']} | {meta['timestamp_s']:.1f}s"
                cv2.putText(
                    annotated, fps_text, (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
                )

                total_frames += 1

                # ── Output video ───────────────────────────────────────────
                if args.output:
                    if output_writer is None:
                        h, w = annotated.shape[:2]
                        output_writer = cv2.VideoWriter(
                            args.output, fourcc, vid.source_fps, (w, h)
                        )
                    output_writer.write(annotated)

                # ── Display ────────────────────────────────────────────────
                if show:
                    cv2.imshow("Road Accident Detection", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("User pressed Q — stopping.")
                        break

                # ── Max-runtime guard ──────────────────────────────────────
                if args.max_seconds > 0:
                    if (time.time() - start_time) > args.max_seconds:
                        logger.info("Max runtime reached.")
                        break

    finally:
        if output_writer:
            output_writer.release()
        if show:
            cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("─" * 50)
    logger.info("Processing complete.")
    logger.info("  Total frames   : %d", total_frames)
    logger.info("  Total accidents: %d", total_accidents)
    logger.info("  Runtime        : %.1f s", elapsed)
    logger.info("  Avg FPS        : %.1f", total_frames / max(elapsed, 1))
    logger.info("  Alerts saved in: %s", args.alerts_dir)
    logger.info("─" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Road Accident Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source
    p.add_argument("--source", default="0",
                   help="Video file path, webcam index, or RTSP URL")
    p.add_argument("--skip", type=int, default=2,
                   help="Process every Nth frame")
    p.add_argument("--max-seconds", type=float, default=0,
                   help="Stop after N seconds (0=unlimited)")

    # Model
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLOv8 model weights file")
    p.add_argument("--conf", type=float, default=0.40,
                   help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.15,
                   help="Collision IoU threshold")
    p.add_argument("--speed-drop", type=float, default=0.20,
                   help="Sudden-stop speed-drop ratio")
    p.add_argument("--cooldown", type=int, default=45,
                   help="Alert cooldown in frames")
    p.add_argument("--device", default="cpu",
                   help="Inference device: cpu or cuda")

    # Alerts
    p.add_argument("--alerts-dir", default="alerts",
                   help="Directory to save screenshots and clips")
    p.add_argument("--clip-pre", type=float, default=5.0,
                   help="Seconds of video before accident to include in clip")
    p.add_argument("--clip-post", type=float, default=5.0,
                   help="Seconds of video after accident to include in clip")
    p.add_argument("--webhook", default="",
                   help="HTTP POST webhook URL for real-time notifications")

    # Output
    p.add_argument("--output", default="",
                   help="Path to save annotated output video")
    p.add_argument("--show", action="store_true", default=True,
                   help="Display live video window")
    p.add_argument("--headless", action="store_true",
                   help="Disable display window (for server / CI use)")

    # Dashboard shortcut
    p.add_argument("--dashboard", action="store_true",
                   help="Launch Streamlit dashboard instead of CLI pipeline")

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.dashboard:
        import subprocess
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        logger.info("Launching Streamlit dashboard…")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            check=True,
        )
    else:
        run_pipeline(args)
