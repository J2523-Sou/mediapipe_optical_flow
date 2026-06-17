"""Small OpenCV video helpers shared by the processing scripts."""

from __future__ import annotations

from pathlib import Path

import cv2


def make_video_writer(cap: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"cannot create output video: {output_path}")
    return writer


def video_timestamp_ms(cap: cv2.VideoCapture, frame_idx: int) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return int((frame_idx / fps) * 1000)
    return frame_idx * 33
