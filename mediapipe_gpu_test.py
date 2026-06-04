#!/usr/bin/env python3
"""Minimal MediaPipe Tasks hand landmark sample.

The script downloads the hand landmark model on first run, tries the GPU
delegate first, and falls back to CPU automatically if GPU is unavailable.

Use:
  python3 mediapipe_gpu_test.py

Quick verification only:
  python3 mediapipe_gpu_test.py --check

Quit the camera window with `q`.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe_tasks_delegate import cpu_delegate_name, explain_cpu_fallback, should_try_gpu

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"


def ensure_model_file() -> Path:
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	if not MODEL_PATH.exists():
		print(f"downloading model: {MODEL_URL}")
		urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
	return MODEL_PATH


def make_options(delegate: mp_tasks.BaseOptions.Delegate) -> vision.HandLandmarkerOptions:
	base_options = mp_tasks.BaseOptions(model_asset_path=str(ensure_model_file()), delegate=delegate)
	return vision.HandLandmarkerOptions(
		base_options=base_options,
		running_mode=vision.RunningMode.VIDEO,
		num_hands=2,
		min_hand_detection_confidence=0.5,
		min_hand_presence_confidence=0.5,
		min_tracking_confidence=0.5,
	)


def create_landmarker(prefer_gpu: bool = True) -> tuple[vision.HandLandmarker, str]:
	explain_cpu_fallback(prefer_gpu)
	if should_try_gpu(prefer_gpu):
		try:
			return vision.HandLandmarker.create_from_options(make_options(mp_tasks.BaseOptions.Delegate.GPU)), "GPU"
		except Exception as exc:
			print(f"GPU delegate unavailable, falling back to CPU: {exc}")
	return vision.HandLandmarker.create_from_options(make_options(mp_tasks.BaseOptions.Delegate.CPU)), cpu_delegate_name()


HAND_CONNECTIONS = [
	(0, 1), (1, 2), (2, 3), (3, 4),
	(0, 5), (5, 6), (6, 7), (7, 8),
	(5, 9), (9, 10), (10, 11), (11, 12),
	(9, 13), (13, 14), (14, 15), (15, 16),
	(13, 17), (17, 18), (18, 19), (19, 20),
	(0, 17),
]


def draw_results(frame, results) -> None:
	if results.hand_landmarks:
		for hand_landmarks in results.hand_landmarks:
			points = []
			for landmark in hand_landmarks:
				height, width = frame.shape[:2]
				x = int(landmark.x * width)
				y = int(landmark.y * height)
				points.append((x, y))
				cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
			for start, end in HAND_CONNECTIONS:
				if start < len(points) and end < len(points):
					cv2.line(frame, points[start], points[end], (255, 0, 0), 2)


def run_camera(prefer_gpu: bool = True) -> None:
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise SystemExit("カメラを開けませんでした。カメラ許可とデバイス番号を確認してください。")

	landmarker, delegate_name = create_landmarker(prefer_gpu=prefer_gpu)
	print(f"delegate: {delegate_name}")
	with landmarker:
		start_time = time.monotonic()
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
			timestamp_ms = int((time.monotonic() - start_time) * 1000)
			results = landmarker.detect_for_video(mp_image, timestamp_ms)
			draw_results(frame, results)
			cv2.imshow("MediaPipe Tasks Hand Landmarker", frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

	cap.release()
	cv2.destroyAllWindows()


def run_check(prefer_gpu: bool = True) -> None:
	landmarker, delegate_name = create_landmarker(prefer_gpu=prefer_gpu)
	with landmarker:
		print(f"mediapipe version: {getattr(mp, '__version__', 'unknown')}")
		print(f"delegate: {delegate_name}")
		print(f"model: {ensure_model_file()}")
		print("tasks API import and model initialization: OK")


def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Minimal MediaPipe Tasks hand landmarker sample")
	parser.add_argument("--check", action="store_true", help="download model and initialize the landmarker only")
	parser.add_argument("--cpu", action="store_true", help="force CPU delegate")
	return parser.parse_args(argv)


def main(argv: list[str]) -> int:
	args = parse_args(argv)
	if args.check:
		run_check(prefer_gpu=not args.cpu)
	else:
		run_camera(prefer_gpu=not args.cpu)
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
