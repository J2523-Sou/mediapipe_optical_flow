#!/usr/bin/env python3
"""リアルタイムにつま先を追跡するプログラム。

MediaPipe Pose は実行時に全身を推定するが、負荷を抑えるため N フレームごと、
または光学フローの追跡失敗時だけ実行する。それ以外のフレームでは、選択した
つま先の周辺クロップだけを OpenCV の光学フローで追跡する。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe_helper import create_pose_landmarker

TOE_LANDMARKS = {
    "left": 31,
    "right": 32,
}


def detect_toe(
    landmarker: vision.PoseLandmarker,
    frame: np.ndarray,
    timestamp_ms: int,
    landmark_index: int,
    pose_scale: float,
) -> tuple[float, float] | None:
    """Pose Landmarker から指定側のつま先をピクセル座標で取得する。"""
    height, width = frame.shape[:2]
    if pose_scale != 1.0:
        pose_frame = cv2.resize(frame, None, fx=pose_scale, fy=pose_scale, interpolation=cv2.INTER_AREA)
    else:
        pose_frame = frame

    frame_rgba = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGBA)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGBA,
        data=np.ascontiguousarray(frame_rgba),
    )
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.pose_landmarks:
        return None

    landmark = result.pose_landmarks[0][landmark_index]
    if landmark.visibility is not None and landmark.visibility < 0.35:
        return None
    return landmark.x * width, landmark.y * height


def crop_bounds(point: tuple[float, float], width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    """点を中心とする追跡クロップを画像範囲内に切り詰めて返す。"""
    x, y = point
    half = crop_size // 2
    x1 = max(0, int(x) - half)
    y1 = max(0, int(y) - half)
    x2 = min(width, int(x) + half)
    y2 = min(height, int(y) + half)
    return x1, y1, x2, y2


def track_toe(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    point: tuple[float, float],
    crop_size: int,
) -> tuple[float, float] | None:
    """前フレームの点を Lucas-Kanade 法で現フレームへ移動させる。"""
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = crop_bounds(point, width, height, crop_size)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None

    prev_crop = prev_gray[y1:y2, x1:x2]
    crop = gray[y1:y2, x1:x2]
    prev_point = np.array([[[point[0] - x1, point[1] - y1]]], dtype=np.float32)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_crop,
        crop,
        prev_point,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if status is None or status[0][0] != 1:
        return None

    next_x = float(next_points[0][0][0] + x1)
    next_y = float(next_points[0][0][1] + y1)
    if not (0 <= next_x < width and 0 <= next_y < height):
        return None
    return next_x, next_y


def timestamp_for_frame(cap: cv2.VideoCapture, frame_idx: int, start_time: float) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return int((frame_idx / fps) * 1000)
    return int((time.monotonic() - start_time) * 1000)


def draw_overlay(
    frame: np.ndarray,
    point: tuple[float, float] | None,
    crop_size: int,
    delegate: str,
    source: str,
) -> None:
    if point is not None:
        x, y = point
        x1, y1, x2, y2 = crop_bounds(point, frame.shape[1], frame.shape[0], crop_size)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 180, 255), 2)
        cv2.drawMarker(
            frame,
            (int(x), int(y)),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
    cv2.putText(
        frame,
        f"delegate={delegate} source={source}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> int:
    cap_source: int | str = args.camera if args.video is None else args.video
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise SystemExit(f"入力を開けませんでした: {cap_source}")

    csv_file = None
    writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "x", "y", "source"])

    landmarker, delegate = create_pose_landmarker()
    landmark_index = TOE_LANDMARKS[args.side]
    prev_gray = None
    point = None
    source = "none"
    start_time = time.monotonic()
    frame_idx = 0

    try:
        with landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                timestamp_ms = timestamp_for_frame(cap, frame_idx, start_time)
                needs_pose = point is None or frame_idx % args.detect_every == 0

                if not needs_pose and prev_gray is not None:
                    tracked = track_toe(prev_gray, gray, point, args.crop_size)
                    if tracked is not None:
                        point = tracked
                        source = "track"
                    else:
                        needs_pose = True

                if needs_pose:
                    detected = detect_toe(
                        landmarker,
                        frame,
                        timestamp_ms,
                        landmark_index,
                        args.pose_scale,
                    )
                    if detected is not None:
                        point = detected
                        source = "pose"
                    else:
                        point = None
                        source = "none"

                if writer is not None:
                    if point is None:
                        writer.writerow([frame_idx, "", "", source])
                    else:
                        writer.writerow([frame_idx, f"{point[0]:.6f}", f"{point[1]:.6f}", source])

                draw_overlay(frame, point, args.crop_size, delegate, source)
                cv2.imshow("Realtime Toe Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                prev_gray = gray
                frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime toe-only tracker")
    parser.add_argument("--video", help="video file path. If omitted, the camera is used.")
    parser.add_argument("--camera", type=int, default=0, help="camera index when --video is omitted")
    parser.add_argument("--side", choices=sorted(TOE_LANDMARKS), default="left", help="toe landmark to track")
    parser.add_argument("--cpu", action="store_true", help="kept for compatibility; CPU is always used")
    parser.add_argument("--detect-every", type=int, default=15, help="run MediaPipe Pose every N frames")
    parser.add_argument("--crop-size", type=int, default=160, help="tracking crop size around the toe in pixels")
    parser.add_argument("--pose-scale", type=float, default=0.5, help="downscale factor for Pose detection")
    parser.add_argument("--csv", help="optional CSV output path")
    args = parser.parse_args(argv)
    if args.detect_every < 1:
        parser.error("--detect-every must be >= 1")
    if args.crop_size < 32:
        parser.error("--crop-size must be >= 32")
    if not (0.1 <= args.pose_scale <= 1.0):
        parser.error("--pose-scale must be between 0.1 and 1.0")
    return args


def main(argv: list[str]) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
