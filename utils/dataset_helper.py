"""
Utility: Dataset Helper
Helps locate and prepare accident detection datasets for training/testing.

Supported dataset sources:
  1. Kaggle  – Road Accident Detection / Car Crash datasets
  2. Local   – any folder of images or video clips
  3. YouTube – download test videos via yt-dlp
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Kaggle dataset download
# ──────────────────────────────────────────────────────────────────────────────

KAGGLE_DATASETS = {
    "road-accident-detection": "ckay16/accident-detection-from-cctv-footage",
    "car-crash-dataset":       "asefjamilajwad/car-crash-dataset",
    "accident-images":         "hendrickemanuel/road-accidents-dataset",
}


def download_kaggle_dataset(
    dataset_key: str,
    output_dir: str = "datasets",
    unzip: bool = True,
) -> Path:
    """
    Download a Kaggle dataset by its shorthand key.

    Requires:
        pip install kaggle
        ~/.kaggle/kaggle.json with your API credentials

    Args:
        dataset_key: One of the keys in KAGGLE_DATASETS (or a full kaggle path).
        output_dir:  Directory to save the dataset.
        unzip:       Extract the downloaded zip.

    Returns:
        Path to the downloaded (and optionally unzipped) dataset.
    """
    slug = KAGGLE_DATASETS.get(dataset_key, dataset_key)
    out = Path(output_dir) / dataset_key
    out.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out)]
    if unzip:
        cmd.append("--unzip")

    logger.info("Downloading Kaggle dataset '%s'…", slug)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle download failed:\n{result.stderr}\n\n"
            "Make sure you have:\n"
            "  1. pip install kaggle\n"
            "  2. ~/.kaggle/kaggle.json with your API key"
        )

    logger.info("Dataset saved to %s", out)
    return out


def list_available_datasets() -> dict[str, str]:
    """Return shorthand names and Kaggle slugs for all known datasets."""
    return dict(KAGGLE_DATASETS)


# ──────────────────────────────────────────────────────────────────────────────
# YouTube test-video downloader (yt-dlp)
# ──────────────────────────────────────────────────────────────────────────────

# Public domain / creative-commons dashboard footage useful for testing
SAMPLE_VIDEO_URLS = [
    "https://www.youtube.com/watch?v=MNn9qKG2UFI",   # dashcam compilation (CC)
    "https://www.youtube.com/watch?v=wqctLW0Hb_0",   # traffic cam footage (CC)
]


def download_youtube_video(
    url: str,
    output_dir: str = "test_videos",
    max_duration_s: int = 120,
) -> Path:
    """
    Download a YouTube video for testing purposes using yt-dlp.

    Requires:
        pip install yt-dlp

    Args:
        url:            YouTube watch URL.
        output_dir:     Where to save the mp4.
        max_duration_s: Skip download if video > this many seconds.

    Returns:
        Path to the downloaded mp4.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed: pip install yt-dlp")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out / "%(title)s.%(ext)s"),
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration < {max_duration_s}"
        ),
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    logger.info("Video downloaded: %s", filename)
    return Path(filename)


# ──────────────────────────────────────────────────────────────────────────────
# Local dataset scanner
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def scan_local_dataset(root: str) -> dict:
    """
    Scan a local directory and return a summary of image and video files.

    Returns:
        {
          "images": [list of Path],
          "videos": [list of Path],
          "total_files": int,
        }
    """
    root_path = Path(root)
    images, videos = [], []

    for p in root_path.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
        elif p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)

    summary = {
        "images": images,
        "videos": videos,
        "total_files": len(images) + len(videos),
    }

    logger.info(
        "Dataset scan: %d images, %d videos in '%s'",
        len(images), len(videos), root,
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# YOLO dataset formatter (for custom training)
# ──────────────────────────────────────────────────────────────────────────────

def create_yolo_dataset_yaml(
    dataset_dir: str,
    class_names: list[str] | None = None,
    output_path: str = "dataset.yaml",
) -> Path:
    """
    Generate a dataset.yaml file for YOLOv8 training.

    Args:
        dataset_dir:  Root of the dataset (must contain train/ and val/ sub-dirs).
        class_names:  List of class labels (default: accident / no-accident).
        output_path:  Where to write the yaml.
    """
    if class_names is None:
        class_names = ["no-accident", "accident"]

    yaml_content = (
        f"path: {Path(dataset_dir).resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    out = Path(output_path)
    out.write_text(yaml_content)
    logger.info("Dataset YAML written to %s", out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Dataset utility for accident detection")
    sub = p.add_subparsers(dest="command")

    dl = sub.add_parser("download-kaggle", help="Download a Kaggle dataset")
    dl.add_argument("key", choices=list(KAGGLE_DATASETS), help="Dataset shorthand")
    dl.add_argument("--out", default="datasets")

    yt = sub.add_parser("download-youtube", help="Download a YouTube test video")
    yt.add_argument("url", help="YouTube video URL")
    yt.add_argument("--out", default="test_videos")

    sc = sub.add_parser("scan", help="Scan a local directory")
    sc.add_argument("path", help="Directory to scan")

    ls = sub.add_parser("list", help="List available Kaggle datasets")

    args = p.parse_args()

    if args.command == "download-kaggle":
        download_kaggle_dataset(args.key, output_dir=args.out)

    elif args.command == "download-youtube":
        download_youtube_video(args.url, output_dir=args.out)

    elif args.command == "scan":
        result = scan_local_dataset(args.path)
        print(json.dumps(
            {k: [str(v) for v in vals] if isinstance(vals, list) else vals
             for k, vals in result.items()},
            indent=2,
        ))

    elif args.command == "list":
        for k, v in list_available_datasets().items():
            print(f"  {k:<30} → {v}")

    else:
        p.print_help()
