"""朴素基线（persistence / seasonal-naive），跑在与阶段 06 相同的划分上。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as dp


INPUT_WINDOW = 12
OUTPUT_WINDOW = 12
TRAIN_RATE = 0.6
EVAL_RATE = 0.2
HORIZON_STEPS = (1, 3, 12)   # 与 06 的 output_windows 对齐


def read_dyna_matrix(dyna_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """把 .dyna 读成 (时间戳列表, 摄像头列表, 矩阵[时间, 摄像头])。"""
    if not dyna_path.is_file():
        raise FileNotFoundError(f"LibCity dyna file not found: {dyna_path}")
    values: dict[tuple[str, str], float] = {}
    times: list[str] = []
    seen_times = set()
    cameras: set[str] = set()
    with dyna_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            time = row["time"]
            entity = row["entity_id"]
            if time not in seen_times:
                seen_times.add(time)
                times.append(time)
            cameras.add(entity)
            try:
                values[(time, entity)] = float(row.get("total_vehicles", 0.0))
            except (TypeError, ValueError):
                values[(time, entity)] = 0.0
    times.sort()
    camera_list = sorted(cameras)
    matrix = np.array(
        [[values.get((t, c), 0.0) for c in camera_list] for t in times],
        dtype=np.float64,
    )
    return times, camera_list, matrix


def infer_step_minutes(times: list[str]) -> int:
    """从时间戳序列推断采样间隔分钟数，用于算「前一天同一时刻」的滞后步数。"""
    if len(times) < 3:
        return 10
    stamps = []
    for t in times[:200]:
        try:
            stamps.append(datetime.fromisoformat(t.replace("Z", "+00:00")))
        except ValueError:
            continue
    stamps.sort()
    gaps = sorted((b - a).total_seconds() / 60.0 for a, b in zip(stamps, stamps[1:]) if b > a)
    if not gaps:
        return 10
    return max(1, int(round(gaps[len(gaps) // 2])))


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """与 `06` 一致的口径：MAE / MSE / RMSE / R2。"""
    difference = prediction - truth
    mse = float(np.mean(difference**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(difference))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - np.sum(difference**2) / max(ss_tot, 1e-12)),
    }


def build_samples(matrix: np.ndarray):
    """按 LibCity 的方式切样本，测试集取最后 20% 个样本。"""
    total = matrix.shape[0] - INPUT_WINDOW - OUTPUT_WINDOW + 1
    if total <= 0:
        raise SystemExit(f"Not enough time steps: {matrix.shape[0]}")
    test_count = round(total * EVAL_RATE)
    test_start = total - test_count
    index = test_start  # 只评估测试段，与 06 报的测试指标可比
    inputs = np.stack([matrix[i : i + INPUT_WINDOW] for i in range(index, total)])
    targets = np.stack([matrix[i + INPUT_WINDOW : i + INPUT_WINDOW + OUTPUT_WINDOW] for i in range(index, total)])
    return inputs, targets, index


def seasonal_lag(times: list[str], step_minutes: int) -> int:
    """「前一天同一时刻」对应的步数。采样间隔变化时自动跟着变。"""
    return max(1, int(round(24 * 60 / step_minutes)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run naive forecasting baselines on the same split as stage 06")
    dp.add_tag_argument(parser)
    parser.add_argument("--dyna", type=Path, default=None, help="Default: results/<tag>/05_libcity_dataset/isy5002_traffic.dyna")
    parser.add_argument("--output", type=Path, default=None, help="Default: results/<tag>/06b_baselines")
    args = parser.parse_args()

    tag = dp.resolve_tag(args.dataset_tag)
    dyna_path = args.dyna or dp.stage_dir(tag, "05_libcity_dataset") / "isy5002_traffic.dyna"
    output_dir = args.output or dp.stage_dir(tag, "06b_baselines")
    print(f"Dataset tag: {tag}")

    times, cameras, matrix = read_dyna_matrix(dyna_path)
    step_minutes = infer_step_minutes(times)
    lag = seasonal_lag(times, step_minutes)
    print(f"  {matrix.shape[0]} 个时间点 × {len(cameras)} 个摄像头，间隔 {step_minutes} 分钟")
    print(f"  seasonal_naive 滞后 = {lag} 步（前一天同一时刻）")

    inputs, targets, test_start = build_samples(matrix)
    print(f"  测试样本 {inputs.shape[0]} 个（起点索引 {test_start}）")

    # persistence：用输入窗口最后一个观测值填满整个输出窗口
    persistence = np.repeat(inputs[:, -1:, :], OUTPUT_WINDOW, axis=1)

    # seasonal_naive：取「前一天同一时刻」。测试段起点之前的时刻都可用，不会用到未来。
    seasonal = np.zeros_like(targets)
    for k in range(OUTPUT_WINDOW):
        for i in range(inputs.shape[0]):
            t_target = test_start + i + INPUT_WINDOW + k
            t_source = t_target - lag
            seasonal[i, k] = matrix[t_source] if 0 <= t_source < matrix.shape[0] else inputs[i, -1]

    rows = []
    for name, prediction in (("persistence", persistence), ("seasonal_naive", seasonal)):
        for step in HORIZON_STEPS:
            index = step - 1
            if index >= OUTPUT_WINDOW:
                continue
            pred_step = prediction[:, index, :]
            truth_step = targets[:, index, :]
            row = {
                "model": name,
                "horizon_minutes": step * step_minutes,
                "camera_count": len(cameras),
                **{k: round(v, 6) for k, v in metrics(pred_step, truth_step).items()},
            }
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = output_dir / "baseline_metrics.csv"
    with comparison.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 逐摄像头，便于与 08 的 camera_error_summary 对照
    per_camera = []
    for camera_index, camera_id in enumerate(cameras):
        for name, prediction in (("persistence", persistence), ("seasonal_naive", seasonal)):
            for step in HORIZON_STEPS:
                index = step - 1
                if index >= OUTPUT_WINDOW:
                    continue
                m = metrics(prediction[:, index, camera_index], targets[:, index, camera_index])
                per_camera.append({
                    "camera_id": camera_id, "model": name,
                    "horizon_minutes": step * step_minutes,
                    **{k: round(v, 6) for k, v in m.items()},
                })
    with (output_dir / "baseline_camera_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_camera[0]))
        writer.writeheader()
        writer.writerows(per_camera)

    (output_dir / "run_metadata.json").write_text(
        json.dumps({
            "dataset_tag": tag,
            "step_minutes": step_minutes,
            "seasonal_lag_steps": lag,
            "test_samples": int(inputs.shape[0]),
            "input_window": INPUT_WINDOW,
            "output_window": OUTPUT_WINDOW,
            "cameras": cameras,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'baseline':16s} {'horizon':>8s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}")
    for row in rows:
        print(f"{row['model']:16s} {row['horizon_minutes']:>6d}min {row['MAE']:>8.3f} "
              f"{row['RMSE']:>8.3f} {row['R2']:>+8.3f}")
    print(f"\nSaved: {comparison}")


if __name__ == "__main__":
    main()
