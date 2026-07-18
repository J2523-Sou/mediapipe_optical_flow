#!/usr/bin/env python3
"""座標 CSV から移動方向（軌跡の heading）の角速度を計算する。

画像座標の y 軸は下向きなので、計算時だけ y を反転して数学座標系に直す。
その後、移動方向の unwrap と Savitzky-Golay 微分を行い、ノイズを抑えた
角速度を CSV に書き出す。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


def coordinate_pairs(fieldnames: list[str]) -> list[tuple[str, str, str]]:
    """CSV ヘッダーから対応する x/y 列を探し、(ラベル, x列, y列) を返す。"""
    fields = set(fieldnames)
    pairs = []
    for x_column in fieldnames:
        match = re.fullmatch(r"x(.*)", x_column)
        if not match:
            continue
        suffix = match.group(1)
        y_column = f"y{suffix}"
        if y_column in fields:
            pairs.append((suffix.removeprefix("_") or "raw", x_column, y_column))
    return pairs


def trajectory_angular_velocity(
    frames: np.ndarray,
    points: np.ndarray,
    min_speed: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """heading、未平滑/平滑角速度、速度をフレーム単位で返す。

    速度が極端に小さいフレームでは移動方向が不定になるため、結果を NaN
    にして CSV 上では空欄として扱えるようにする。
    """
    frame_steps = np.diff(frames)
    if not np.allclose(frame_steps, frame_steps[0]):
        raise ValueError("Savitzky-Golay angular velocity requires equally spaced frames")
    frame_step = float(frame_steps[0])
    dx = np.gradient(points[:, 0], frames)
    # Image y increases downward. Negate dy so positive angles are counterclockwise.
    dy = np.gradient(-points[:, 1], frames)
    speed = np.hypot(dx, dy)
    heading = np.unwrap(np.arctan2(dy, dx))
    angular_velocity = np.gradient(heading, frames)
    angular_velocity_savgol = savgol_filter(
        heading,
        window_length=savgol_window,
        polyorder=savgol_polyorder,
        deriv=1,
        delta=frame_step,
    )
    invalid = speed < min_speed
    heading[invalid] = np.nan
    angular_velocity[invalid] = np.nan
    angular_velocity_savgol[invalid] = np.nan
    return heading, angular_velocity, angular_velocity_savgol, speed


def format_value(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.9f}"


def process_csv(
    input_path: Path,
    output_path: Path,
    fps: float | None,
    min_speed: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> None:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "frame" not in fieldnames:
        raise SystemExit("input CSV must contain a frame column")

    pairs = coordinate_pairs(fieldnames)
    if not pairs:
        raise SystemExit("input CSV must contain at least one x/y coordinate pair")

    frames = np.asarray([float(row["frame"]) for row in rows], dtype=np.float64)
    if savgol_window > len(frames):
        raise SystemExit(f"--omega-savgol-window must be <= the number of rows ({len(frames)})")

    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, x_column, y_column in pairs:
        if any(not row[x_column] or not row[y_column] for row in rows):
            print(f"skip {label}: missing coordinates", file=sys.stderr)
            continue
        points = np.asarray(
            [[float(row[x_column]), float(row[y_column])] for row in rows],
            dtype=np.float64,
        )
        results[label] = trajectory_angular_velocity(
            frames,
            points,
            min_speed,
            savgol_window,
            savgol_polyorder,
        )

    headers = ["frame"]
    for label in results:
        headers.extend([
            f"heading_deg_{label}",
            f"omega_deg_per_frame_{label}",
            f"omega_rad_per_frame_{label}",
            f"omega_savgol_deg_per_frame_{label}",
            f"omega_savgol_rad_per_frame_{label}",
            f"speed_px_per_frame_{label}",
        ])
        if fps is not None:
            headers.extend([
                f"omega_deg_per_s_{label}",
                f"omega_rad_per_s_{label}",
                f"omega_savgol_deg_per_s_{label}",
                f"omega_savgol_rad_per_s_{label}",
            ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for index, frame in enumerate(frames):
            output = [format_value(frame)]
            for heading, omega, omega_savgol, speed in results.values():
                output.extend([
                    format_value(np.degrees(heading[index])),
                    format_value(np.degrees(omega[index])),
                    format_value(omega[index]),
                    format_value(np.degrees(omega_savgol[index])),
                    format_value(omega_savgol[index]),
                    format_value(speed[index]),
                ])
                if fps is not None:
                    output.extend([
                        format_value(np.degrees(omega[index]) * fps),
                        format_value(omega[index] * fps),
                        format_value(np.degrees(omega_savgol[index]) * fps),
                        format_value(omega_savgol[index] * fps),
                    ])
            writer.writerow(output)

    print(f"saved: {output_path}")
    print(f"coordinate sets: {', '.join(results)}")
    print("angular velocity: trajectory heading change, positive=counterclockwise")
    print(f"angular velocity Savitzky-Golay: window={savgol_window}, polyorder={savgol_polyorder}")
    if fps is None:
        print("FPS was not specified; output units are per frame.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate trajectory-heading angular velocity from CSV")
    parser.add_argument("input", type=Path, help="CSV containing frame and x/y coordinate pairs")
    parser.add_argument("-o", "--output", type=Path, help="output CSV path")
    parser.add_argument("--fps", type=float, help="video FPS; enables per-second angular velocity columns")
    parser.add_argument(
        "--omega-savgol-window",
        type=int,
        default=11,
        help="Savitzky-Golay window for angular velocity differentiation",
    )
    parser.add_argument(
        "--omega-savgol-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order for angular velocity differentiation",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=1e-6,
        help="set heading/angular velocity blank below this speed in px/frame",
    )
    args = parser.parse_args(argv)
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.min_speed < 0:
        parser.error("--min-speed must be >= 0")
    if args.omega_savgol_window < 3 or args.omega_savgol_window % 2 == 0:
        parser.error("--omega-savgol-window must be an odd integer >= 3")
    if (
        args.omega_savgol_polyorder < 1
        or args.omega_savgol_polyorder >= args.omega_savgol_window
    ):
        parser.error(
            "--omega-savgol-polyorder must be >= 1 and less than --omega-savgol-window"
        )
    if args.output is None:
        args.output = args.input.with_name(f"{args.input.stem}_angular_velocity.csv")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    process_csv(
        args.input,
        args.output,
        args.fps,
        args.min_speed,
        args.omega_savgol_window,
        args.omega_savgol_polyorder,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
