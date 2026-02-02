from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def extract_audio(video_path: str | Path, output_audio_path: str | Path) -> bool:
    ensure_parent(output_audio_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def extract_frame(video_path: str | Path, output_image_path: str | Path, timestamp: Optional[float] = None) -> bool:
    ensure_parent(output_image_path)
    if timestamp is None:
        timestamp = _get_middle_timestamp(video_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        str(timestamp),
        "-vframes",
        "1",
        str(output_image_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _get_middle_timestamp(video_path: str | Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        return duration / 2
    except (subprocess.CalledProcessError, ValueError):
        return 1.0


def cleanup_paths(*paths: str | Path) -> None:
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
