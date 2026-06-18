"""CPU-only MediaPipe Tasks helpers."""

from __future__ import annotations

import platform
import urllib.request
from pathlib import Path

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


MODEL_CACHE_DIR = Path.home() / ".cache" / "mediapipe"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = MODEL_CACHE_DIR / "hand_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
POSE_MODEL_PATH = MODEL_CACHE_DIR / "pose_landmarker_full.task"


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def cpu_delegate_name() -> str:
    if is_apple_silicon():
        return "CPU (Apple Silicon)"
    return "CPU"


def ensure_model_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"downloading model: {url}")
        urllib.request.urlretrieve(url, path)
    return path


def ensure_hand_model_file() -> Path:
    return ensure_model_file(HAND_MODEL_URL, HAND_MODEL_PATH)


def ensure_pose_model_file() -> Path:
    return ensure_model_file(POSE_MODEL_URL, POSE_MODEL_PATH)


def cpu_base_options(model_path: Path) -> mp_tasks.BaseOptions:
    return mp_tasks.BaseOptions(
        model_asset_path=str(model_path),
        delegate=mp_tasks.BaseOptions.Delegate.CPU,
    )


def make_hand_options() -> vision.HandLandmarkerOptions:
    return vision.HandLandmarkerOptions(
        base_options=cpu_base_options(ensure_hand_model_file()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def create_hand_landmarker() -> tuple[vision.HandLandmarker, str]:
    return vision.HandLandmarker.create_from_options(make_hand_options()), cpu_delegate_name()


def make_pose_options(
    *,
    output_segmentation_masks: bool = False,
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> vision.PoseLandmarkerOptions:
    return vision.PoseLandmarkerOptions(
        base_options=cpu_base_options(ensure_pose_model_file()),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=output_segmentation_masks,
    )


def create_pose_landmarker(
    *,
    output_segmentation_masks: bool = False,
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[vision.PoseLandmarker, str]:
    return (
        vision.PoseLandmarker.create_from_options(
            make_pose_options(
                output_segmentation_masks=output_segmentation_masks,
                num_poses=num_poses,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        ),
        cpu_delegate_name(),
    )
