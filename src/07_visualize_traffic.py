"""Visualize LibCity traffic prediction results from the latest experiment batch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dataset_paths as dp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("fnn", "gru", "stgcn")
MODEL_LABELS = {"fnn": "FNN", "gru": "GRU", "stgcn": "STGCN"}


def find_latest_batch(experiment_root: Path) -> Path:
    """查找最新的完整实验批次（要求含 model_comparison.csv，避免取到跑空的批次）。"""
    batches = sorted(experiment_root.glob("run_*"), key=lambda path: path.name)
    if not batches:
        raise FileNotFoundError(f"No experiment batch found: {experiment_root}")
    complete = [path for path in batches if (path / "model_comparison.csv").is_file()]
    if not complete:
        raise FileNotFoundError(
            f"No complete experiment batch under {experiment_root}: "
            f"none of {len(batches)} run_* directories contains model_comparison.csv"
        )
    return complete[-1]


def read_experiment_config(batch_dir: Path) -> dict:
    """读取批次配置；文件缺失或损坏时返回空字典。"""
    config_path = batch_dir / "experiment_config.json"
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def interval_minutes_of(config: dict) -> int:
    """采样间隔；旧批次没有该字段时回退到 5 分钟。"""
    try:
        interval = int(config.get("interval_minutes", 5))
    except (TypeError, ValueError):
        return 5
    return interval if interval > 0 else 5


def horizon_steps(config: dict, legacy_horizon: str | None) -> list[tuple[str, int]]:
    """返回 [(跨度标签, 预测步序号)]，标签随采样间隔变化。"""
    interval = interval_minutes_of(config)
    if legacy_horizon:
        try:
            minutes = int(legacy_horizon.replace("min", ""))
        except ValueError:
            return []
        return [(legacy_horizon, max(0, minutes // interval - 1))]
    windows = config.get("output_windows")
    steps: list[tuple[str, int]] = []
    if isinstance(windows, (list, tuple)):
        for window in windows:
            try:
                step = int(window)
            except (TypeError, ValueError):
                continue
            if step > 0:
                steps.append((f"{step * interval}min", step - 1))
    if steps:
        return steps
    return [(label, max(0, int(label.replace("min", "")) // interval - 1)) for label in ("5min", "15min", "60min")]


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 文件。"""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_timestamps(dyna_file: Path) -> list[str]:
    """读取数据集中的唯一时间戳。"""
    if not dyna_file.is_file():
        return []
    timestamps = []
    seen = set()
    with dyna_file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            value = row.get("time", "")
            if value and value not in seen:
                timestamps.append(value)
                seen.add(value)
    return timestamps


def read_camera_ids(geo_file: Path) -> list[str]:
    """读取 geo 文件中的摄像头顺序。"""
    if not geo_file.is_file():
        return []
    with geo_file.open(newline="", encoding="utf-8") as stream:
        return [row.get("geo_id", "") for row in csv.DictReader(stream)]


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取并规范化预测数组。"""
    payload = np.load(path)
    prediction = np.asarray(payload["prediction"], dtype=float)
    truth = np.asarray(payload["truth"], dtype=float)
    if prediction.ndim == 4:
        prediction = prediction[..., 0]
    if truth.ndim == 4:
        truth = truth[..., 0]
    if prediction.ndim != 3 or truth.shape != prediction.shape:
        raise ValueError(f"Unexpected prediction shape in {path}: {prediction.shape}, {truth.shape}")
    return prediction, truth


def sample_labels(sample_count: int, horizon: int, input_window: int, timestamps: list[str], output_window: int = 12) -> list[str]:
    """将测试样本索引映射到预测目标时间标签。"""
    # LibCity 按完整输出窗口生成样本，并采用 round 划分测试集。
    sample_total = max(0, len(timestamps) - input_window - output_window + 1)
    test_count = round(sample_total * 0.2)
    test_start = sample_total - test_count
    labels = []
    for index in range(sample_count):
        # horizon 表示从当前输入末端起的预测步数，第一步为 t+1。
        # input_window 已经比 LibCity 的样本末端索引多 1，因此减去 1。
        timestamp_index = input_window + test_start + index + horizon - 1
        labels.append(timestamps[timestamp_index] if timestamp_index < len(timestamps) else str(index + 1))
    return labels


def plot_metrics(batch_dir: Path, rows: list[dict[str, str]], output_dir: Path) -> None:
    """绘制不同模型和预测时长的总体指标。"""
    if not rows:
        return
    horizons = sorted({row.get("horizon_minutes", "") for row in rows}, key=lambda value: float(value or 0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, title in zip(axes, ("MAE", "RMSE", "R2"), ("MAE by Horizon", "RMSE by Horizon", "R-squared by Horizon")):
        width = 0.24
        x = np.arange(len(horizons))
        for model_index, model in enumerate(("FNN", "GRU", "STGCN")):
            values = []
            for horizon in horizons:
                match = next((row for row in rows if row.get("model") == model and row.get("horizon_minutes") == horizon), None)
                try:
                    values.append(float(match.get(metric, "nan")) if match else np.nan)
                except ValueError:
                    values.append(np.nan)
            axis.bar(x + (model_index - 1) * width, values, width=width, label=model)
        axis.set_title(title)
        axis.set_xticks(x, [f"{value} min" for value in horizons])
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_metrics_comparison.png", dpi=150)
    plt.close(fig)


def plot_predictions(batch_dir: Path, output_dir: Path, camera_ids: list[str], dyna_file: Path) -> None:
    """为每个模型和预测时长绘制摄像头级实际值与预测值。"""
    config = read_experiment_config(batch_dir)
    input_window = int(config.get("input_window", 12))
    timestamps = read_timestamps(dyna_file)
    model_dirs = [(model_dir, None) for model_dir in sorted((batch_dir / "models").glob("*")) if model_dir.is_dir()]
    if not model_dirs:
        model_dirs = [(horizon_dir / model, horizon_dir.name.replace("horizon_", "")) for horizon_dir in sorted(batch_dir.glob("horizon_*")) for model in MODEL_ORDER if (horizon_dir / model).is_dir()]
    for model_dir, legacy_horizon in model_dirs:
        prediction_path = model_dir / "predictions.npz"
        if not prediction_path.is_file():
            continue
        prediction, truth = load_predictions(prediction_path)
        for horizon_label, step_index in horizon_steps(config, legacy_horizon):
            if step_index >= prediction.shape[1]:
                continue
            step_prediction = prediction[:, step_index, :]
            step_truth = truth[:, step_index, :]
            camera_count = step_prediction.shape[1]
            # 横轴使用测试样本序号，避免时间标签在不完整数据时产生误导
            x = np.arange(prediction.shape[0])
            rows = int(np.ceil(camera_count / 2))
            fig, axes = plt.subplots(rows, 2, figsize=(14, max(4, rows * 3.2)), squeeze=False)
            for camera_index in range(camera_count):
                axis = axes[camera_index // 2][camera_index % 2]
                axis.plot(x, step_truth[:, camera_index], label="Actual", linewidth=1.2)
                axis.plot(x, step_prediction[:, camera_index], label="Predicted", linewidth=1.1)
                camera_label = camera_ids[camera_index] if camera_index < len(camera_ids) else str(camera_index + 1)
                axis.set_title(f"Camera {camera_label}")
                axis.set_xlabel("Test sample")
                axis.set_ylabel("Vehicles")
                axis.grid(alpha=0.25)
            for unused in range(camera_count, rows * 2):
                axes[unused // 2][unused % 2].axis("off")
            axes[0][0].legend()
            fig.suptitle(f"{MODEL_LABELS[model_dir.name]} - {horizon_label} Prediction")
            fig.tight_layout()
            fig.savefig(output_dir / f"{model_dir.name}_{horizon_label}_predictions.png", dpi=150)
            plt.close(fig)


def plot_camera_errors(batch_dir: Path, output_dir: Path, camera_ids: list[str]) -> None:
    """绘制各摄像头的 MAE 热力图。"""
    config = read_experiment_config(batch_dir)
    records = []
    model_dirs = [(model_dir, None) for model_dir in sorted((batch_dir / "models").glob("*")) if model_dir.is_dir()]
    if not model_dirs:
        model_dirs = [(horizon_dir / model, horizon_dir.name.replace("horizon_", "")) for horizon_dir in sorted(batch_dir.glob("horizon_*")) for model in MODEL_ORDER if (horizon_dir / model).is_dir()]
    for model_dir, legacy_horizon in model_dirs:
        prediction_path = model_dir / "predictions.npz"
        if not prediction_path.is_file():
            continue
        prediction, truth = load_predictions(prediction_path)
        for horizon_label, step_index in horizon_steps(config, legacy_horizon):
            if step_index >= prediction.shape[1]:
                continue
            errors = np.mean(np.abs(prediction[:, step_index, :] - truth[:, step_index, :]), axis=0)
            records.append((horizon_label, MODEL_LABELS.get(model_dir.name, model_dir.name.upper()), errors))
    if not records:
        return
    fig, axes = plt.subplots(len(records), 1, figsize=(10, max(4, len(records) * 1.2)), squeeze=False)
    for row_index, (horizon, model, errors) in enumerate(records):
        axis = axes[row_index][0]
        axis.bar(np.arange(len(errors)), errors, color="#4c78a8")
        axis.set_title(f"{model} - {horizon} Camera MAE")
        axis.set_ylabel("MAE")
        labels = [camera_ids[index] if index < len(camera_ids) else str(index + 1) for index in range(len(errors))]
        axis.set_xticks(np.arange(len(errors)), labels)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "camera_error_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """生成最新实验批次的全部图表。"""
    parser = argparse.ArgumentParser(description="Visualize LibCity prediction results for one dataset tag")
    dp.add_tag_argument(parser)
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag, interactive=True)

    experiment_root = dp.stage_dir(tag, "06_libcity_experiment")
    output_root = dp.stage_dir(tag, "07_traffic_visualization")
    dataset_dir = dp.stage_dir(tag, "05_libcity_dataset")
    print(f"Dataset tag: {tag}")

    batch_dir = find_latest_batch(experiment_root)
    print(f"Latest complete batch: {batch_dir}")
    comparison_file = batch_dir / "model_comparison.csv"
    rows = read_csv(comparison_file) if comparison_file.is_file() else []
    output_dir = output_root / batch_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_ids = read_camera_ids(dataset_dir / "isy5002_traffic.geo")
    plot_metrics(batch_dir, rows, output_dir)
    plot_predictions(batch_dir, output_dir, camera_ids, dataset_dir / "isy5002_traffic.dyna")
    plot_camera_errors(batch_dir, output_dir, camera_ids)
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
