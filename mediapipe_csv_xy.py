#!/usr/bin/env python3
"""Export CSV files with the left foot landmark x/y from videos.

This version uses MediaPipe Tasks (0.10.35) with the Pose Landmarker and
tries the GPU delegate first. The input video is decoded with OpenCV, converted
to SRGBA, and passed to the Tasks API in VIDEO mode.
"""

from __future__ import annotations

import argparse
import csv
import os
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
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
	"pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "pose_landmarker_full.task"
RESULTS_DIR = Path("results")

# PoseLandmarker の結果から「左足首」の座標だけを取り出して CSV に出す。
LEFT_FOOT_INDEX = 31


def ensure_model_file() -> Path:
	# モデルは毎回ダウンロードしない。初回だけ取得してキャッシュする。
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	if not MODEL_PATH.exists():
		print(f"downloading model: {MODEL_URL}")
		urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
	return MODEL_PATH


def make_pose_options(delegate: mp_tasks.BaseOptions.Delegate) -> vision.PoseLandmarkerOptions:
	# Tasks API の設定本体。
	# model_asset_path でモデルを指定し、delegate で CPU/GPU を切り替える。
	base_options = mp_tasks.BaseOptions(model_asset_path=str(ensure_model_file()), delegate=delegate)
	return vision.PoseLandmarkerOptions(
		base_options=base_options,
		running_mode=vision.RunningMode.VIDEO,
		# 動画ごとに1人だけを追跡したいので 1 にする。
		num_poses=1,
		# 検出のしきい値。低いほど拾いやすく、高いほど厳しくなる。
		min_pose_detection_confidence=0.5,
		# ランドマークが「本当に存在する」とみなすしきい値。
		min_pose_presence_confidence=0.5,
		# フレーム間追跡のしきい値。低いと追跡しやすいが誤追跡も増えやすい。
		min_tracking_confidence=0.5,
		# このスクリプトではマスクは使わない。
		output_segmentation_masks=False,
	)


def create_pose_landmarker(prefer_gpu: bool = True) -> tuple[vision.PoseLandmarker, str]:
	# まず GPU を試し、だめなら CPU にフォールバックする。
	explain_cpu_fallback(prefer_gpu)
	if should_try_gpu(prefer_gpu):
		try:
			return vision.PoseLandmarker.create_from_options(
				make_pose_options(mp_tasks.BaseOptions.Delegate.GPU)
			), "GPU"
		except Exception as exc:
			print(f"GPU delegate unavailable, falling back to CPU: {exc}")
	# GPU が使えない環境や --cpu 指定時はこちら。
	return vision.PoseLandmarker.create_from_options(
		make_pose_options(mp_tasks.BaseOptions.Delegate.CPU)
	), cpu_delegate_name()


def get_video_timestamp_ms(cap: cv2.VideoCapture, frame_index: int) -> int:
	# VIDEO モードではフレームのタイムスタンプが必要。
	# FPS が取れるなら frame_index / fps から算出し、取れなければ仮の 33ms 刻みにする。
	fps = cap.get(cv2.CAP_PROP_FPS)
	if fps and fps > 0:
		return int((frame_index / fps) * 1000)
	return frame_index * 33


def extract_left_foot_xy(result: vision.PoseLandmarkerResult, width: float, height: float):
	# pose_landmarks が空なら検出なし。
	# 1 人分のランドマーク配列のうち、LEFT_FOOT_INDEX だけ取り出す。
	if not result.pose_landmarks:
		return None
	landmark = result.pose_landmarks[0][LEFT_FOOT_INDEX]
	# MediaPipe の正規化座標をピクセル座標に戻す。
	return landmark.x * width, landmark.y * height


def process_video(video_path: str, landmarker: vision.PoseLandmarker, delegate: str) -> None:
	# 1 本の動画を読み、各フレームの左足首座標を CSV に書き出す。
	print(f"Processing: {video_path}")
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"  -> cannot open: {video_path}")
		return

	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	base = os.path.splitext(os.path.basename(video_path))[0]
	csv_path = RESULTS_DIR / f"{base}.csv"

	with csv_path.open("w", newline="", encoding="utf-8") as f:
		# CSV は frame / x / y の3列だけにする。
		writer = csv.writer(f)
		writer.writerow(["frame", "x", "y"])

		frame_idx = 0
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			# OpenCV の BGR 画像を MediaPipe が扱いやすい RGBA に変換する。
			frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
			# Tasks API は mp.Image を受け取るのでここで包む。
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
			timestamp_ms = get_video_timestamp_ms(cap, frame_idx)
			# VIDEO モードの推論実行。
			result = landmarker.detect_for_video(mp_image, timestamp_ms)
			landmark_xy = extract_left_foot_xy(result, width, height)

			if landmark_xy is None:
				# 検出なしなら空欄で出力する。
				x_value = ""
				y_value = ""
				if frame_idx % 30 == 0:
					print(f"no landmarks at frame={frame_idx}")
			else:
				# 検出ありなら小数6桁で保存する。
				x, y = landmark_xy
				x_value = f"{x:.6f}"
				y_value = f"{y:.6f}"
				if frame_idx % 30 == 0:
					print(f"landmark frame={frame_idx} x={x:.3f} y={y:.3f}")

			# フレーム番号と座標を1行ずつ書く。
			writer.writerow([frame_idx, x_value, y_value])
			print(
				f"now : {frame_idx} / {total_frame} / {x_value} / {y_value} / delegate={delegate} / file {base}.csv"
			)

			frame_idx += 1

	cap.release()
	print(f"  -> saved CSV: {csv_path}")


def select_videos() -> list[str]:
	# Tk のファイルダイアログで複数動画を選択する。
	# この関数を分けておくと、GUI まわりだけ差し替えやすい。
	try:
		import tkinter as tk
		from tkinter import filedialog
	except Exception as exc:  # pragma: no cover - GUI environment dependent
		raise SystemExit(
			"tkinter が使えません。_tkinter 付きの Python を使ってください。"
			f" ({exc})"
		)

	root = tk.Tk()
	root.withdraw()
	# .mp4 などを複数選べるようにしておく。
	video_files = filedialog.askopenfilenames(
		title="解析する動画を選択",
		filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
	)
	root.destroy()
	return list(video_files)


def parse_args(argv: list[str]) -> argparse.Namespace:
	# --cpu を付けると GPU を使わずに動かせる。
	# --video は GUI を使わずにコマンドラインで直接動画を渡したいとき用。
	parser = argparse.ArgumentParser(description="Export LEFT_FOOT_INDEX x/y from videos to CSV")
	parser.add_argument("--cpu", action="store_true", help="force CPU delegate instead of GPU")
	parser.add_argument(
		"--video",
		action="append",
		help="video file path to process (repeatable). If omitted, a tkinter file picker opens.",
	)
	return parser.parse_args(argv)


def main(argv: list[str]) -> int:
	# 実行の流れは、引数解析 -> 動画選択 -> 1 回だけ landmarker を初期化 -> 各動画を処理、の順。
	args = parse_args(argv)
	RESULTS_DIR.mkdir(exist_ok=True)
	videos = args.video if args.video else select_videos()
	if not videos:
		print("No files selected. 終了します。")
		return 0

	# 1 回初期化した landmarker を全動画で使い回す。
	landmarker, delegate = create_pose_landmarker(prefer_gpu=not args.cpu)
	with landmarker:
		print(f"mediapipe version: {getattr(mp, '__version__', 'unknown')}")
		print(f"delegate: {delegate}")
		print(f"model: {ensure_model_file()}")
		# 選択された動画を順番に処理して CSV を作る。
		for video_path in videos:
			process_video(video_path, landmarker=landmarker, delegate=delegate)

	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
