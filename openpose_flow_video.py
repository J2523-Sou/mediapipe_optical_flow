#!/usr/bin/env python3
"""OpenPose 風のキーポイントフローで動画を処理する。

MediaPipe Pose は指定間隔でつま先位置を更新する時だけ使う。
更新の間は、つま先周辺の小さいクロップ内で Lucas-Kanade optical flow
により点を追跡する。
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

RESULTS_DIR = Path("results")

# realtime_toe_tracker.py を import せず、このファイル単体で実行できるようにする。
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "pose_landmarker_full.task"
TOE_LANDMARKS = {
    "left": 31,
    "right": 32,
}


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def is_macos() -> bool:
    return platform.system() == "Darwin"


def should_try_gpu(prefer_gpu: bool) -> bool:
    # macOS では GPU delegate 初期化時に native 側で abort する場合があるため、
    # デフォルトでは CPU を使う。
    return prefer_gpu and not is_macos()


def cpu_delegate_name() -> str:
    if is_apple_silicon():
        return "CPU (Apple Silicon)"
    return "CPU"


def explain_cpu_fallback(prefer_gpu: bool) -> None:
    if prefer_gpu and is_macos():
        print(
            "macOS detected: using CPU delegate. "
            "MediaPipe Tasks GPU delegate can abort the Python process on some macOS environments."
        )


def ensure_model_file() -> Path:
    # Pose Landmarker モデルは初回だけダウンロードし、以後はキャッシュを使う。
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"downloading model: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_pose_options(delegate: mp_tasks.BaseOptions.Delegate) -> vision.PoseLandmarkerOptions:
    base_options = mp_tasks.BaseOptions(model_asset_path=str(ensure_model_file()), delegate=delegate)
    return vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )


def create_pose_landmarker(prefer_gpu: bool = True) -> tuple[vision.PoseLandmarker, str]:
    # GPU が安定して使える環境だけ GPU を試し、それ以外は CPU を使う。
    explain_cpu_fallback(prefer_gpu)
    if should_try_gpu(prefer_gpu):
        try:
            return vision.PoseLandmarker.create_from_options(
                make_pose_options(mp_tasks.BaseOptions.Delegate.GPU)
            ), "GPU"
        except Exception as exc:
            print(f"GPU delegate unavailable, falling back to CPU: {exc}")
    return vision.PoseLandmarker.create_from_options(
        make_pose_options(mp_tasks.BaseOptions.Delegate.CPU)
    ), cpu_delegate_name()


def detect_toe(
    landmarker: vision.PoseLandmarker,
    frame: np.ndarray,
    timestamp_ms: int,
    landmark_index: int,
    pose_scale: float,
    ) -> tuple[float, float] | None:
    # Pose は更新フレームだけ実行する。返ってくる正規化座標は、
    # 元動画の座標系に戻して返す。
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
    # optical flow の追跡範囲を、直前のつま先位置を中心に切り出す。
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
    # 毎フレーム全身 Pose を実行せず、小さいクロップ内でつま先点だけを追跡する。
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


def draw_overlay(
    frame: np.ndarray,
    point: tuple[float, float] | None,
    crop_size: int,
    delegate: str,
    source: str,
) -> None:
    # 注釈付き動画用に、つま先点・crop 範囲・検出元を描画する。
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


def select_videos() -> list[str]:
    # --video が省略された場合はファイル選択ダイアログを使う。
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - GUI environment dependent
        raise SystemExit(
            "tkinter が使えません。--video で動画ファイルを指定してください。"
            f" ({exc})"
        )

    root = tk.Tk()
    root.withdraw()
    video_files = filedialog.askopenfilenames(
        title="OpenPose flow で処理する動画を選択",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
    )
    root.destroy()
    return list(video_files)


def output_paths(video_path: str, out_dir: Path, write_video: bool) -> tuple[Path, Path | None]:
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = out_dir / f"{base}_toe_flow.csv"
    if not write_video:
        return csv_path, None
    return csv_path, out_dir / f"{base}_toe_flow.mp4"


def make_writer(cap: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"出力動画を作成できませんでした: {output_path}")
    return writer


def video_timestamp_ms(cap: cv2.VideoCapture, frame_idx: int) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return int((frame_idx / fps) * 1000)
    return frame_idx * 33


def process_video(args: argparse.Namespace, video_path: str, landmarker, delegate: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open: {video_path}")
        return

    csv_path, annotated_path = output_paths(video_path, args.out_dir, args.write_video)
    video_writer = make_writer(cap, annotated_path) if annotated_path is not None else None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmark_index = TOE_LANDMARKS[args.side]
    prev_gray = None
    point = None
    source = "none"
    frame_idx = 0

    print(f"Processing: {video_path}")
    print(f"  csv: {csv_path}")
    if annotated_path is not None:
        print(f"  annotated video: {annotated_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "source", "crop_x1", "crop_y1", "crop_x2", "crop_y2"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp_ms = video_timestamp_ms(cap, frame_idx)
            # 追跡点がない場合、または指定間隔に達した場合は MediaPipe で再検出する。
            # それ以外の中間フレームは optical flow で追跡する。
            needs_pose = point is None or frame_idx % args.detect_every == 0

            if not needs_pose and prev_gray is not None:
                tracked = track_toe(prev_gray, gray, point, args.crop_size)
                if tracked is not None:
                    point = tracked
                    source = "flow"
                else:
                    # optical flow が点を見失った場合は、同じフレームで Pose に戻して復帰する。
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

            if point is None:
                writer.writerow([frame_idx, "", "", source, "", "", "", ""])
            else:
                x1, y1, x2, y2 = crop_bounds(point, frame.shape[1], frame.shape[0], args.crop_size)
                writer.writerow([
                    frame_idx,
                    f"{point[0]:.6f}",
                    f"{point[1]:.6f}",
                    source,
                    x1,
                    y1,
                    x2,
                    y2,
                ])

            if video_writer is not None:
                draw_overlay(frame, point, args.crop_size, delegate, source)
                video_writer.write(frame)

            if frame_idx % args.progress_every == 0:
                print(f"  frame {frame_idx}/{total_frames} source={source}")

            prev_gray = gray
            frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
    print(f"  done: {frame_idx} frames")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process videos with toe keypoint flow")
    parser.add_argument("--check", action="store_true", help="initialize MediaPipe Tasks and exit")
    parser.add_argument(
        "--video",
        action="append",
        help="video file path to process. Repeat for multiple files. If omitted, a file picker opens.",
    )
    parser.add_argument("--side", choices=sorted(TOE_LANDMARKS), default="left", help="toe landmark to track")
    parser.add_argument("--cpu", action="store_true", help="force CPU delegate")
    parser.add_argument("--detect-every", type=int, default=15, help="run pose detection every N frames")
    parser.add_argument("--crop-size", type=int, default=160, help="optical-flow crop size around the toe")
    parser.add_argument("--pose-scale", type=float, default=0.5, help="downscale factor for pose refresh")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="output directory")
    parser.add_argument("--write-video", action="store_true", help="write an annotated MP4 alongside the CSV")
    parser.add_argument("--progress-every", type=int, default=30, help="print progress every N frames")
    args = parser.parse_args(argv)

    if args.detect_every < 1:
        parser.error("--detect-every must be >= 1")
    if args.crop_size < 32:
        parser.error("--crop-size must be >= 32")
    if not (0.1 <= args.pose_scale <= 1.0):
        parser.error("--pose-scale must be between 0.1 and 1.0")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        landmarker, delegate = create_pose_landmarker(prefer_gpu=not args.cpu)
        with landmarker:
            print(f"delegate: {delegate}")
            print("flow_video MediaPipe Tasks initialization: OK")
        return 0

    videos = args.video if args.video else select_videos()
    if not videos:
        print("No files selected. 終了します。")
        return 0

    landmarker, delegate = create_pose_landmarker(prefer_gpu=not args.cpu)
    with landmarker:
        print(f"delegate: {delegate}")
        for video_path in videos:
            process_video(args, video_path, landmarker, delegate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
