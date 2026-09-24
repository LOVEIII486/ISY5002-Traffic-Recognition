"""Evaluate and summarize all LibCity experiment batches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import dataset_paths as dp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    """读取实验指标 CSV。"""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


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


def collect_results(experiment_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """收集总体指标和摄像头级预测误差。"""
    metric_rows = []
    camera_rows = []
    for batch_dir in sorted(experiment_root.glob("run_*")):
        comparison_file = batch_dir / "model_comparison.csv"
        if not comparison_file.is_file():
            continue
        interval_minutes = interval_minutes_of(read_experiment_config(batch_dir))
        for row in read_rows(comparison_file):
            parsed = {key: row[key] for key in row}
            for key in ("MAE", "RMSE", "R2", "masked_MAE", "masked_RMSE"):
                try:
                    parsed[key] = float(row[key])
                except (KeyError, ValueError):
                    parsed[key] = np.nan
            metric_rows.append(parsed)
        prediction_files = list(batch_dir.glob("models/*/predictions.npz"))
        if not prediction_files:
            prediction_files = list(batch_dir.glob("horizon_*/**/predictions.npz"))
        for prediction_file in prediction_files:
            model_dir = prediction_file.parent
            horizon = model_dir.parent.name.replace("horizon_", "") if model_dir.parent.name.startswith("horizon_") else "multi_horizon"
            model = model_dir.name.upper()
            payload = np.load(prediction_file)
            prediction = np.asarray(payload["prediction"], dtype=float)
            truth = np.asarray(payload["truth"], dtype=float)
            if prediction.ndim == 4:
                prediction = prediction[..., 0]
                truth = truth[..., 0]
            error = np.abs(prediction - truth)
            if error.ndim == 3:
                for horizon_step in (1, 3, 12):
                    if horizon_step <= error.shape[1]:
                        for camera_index, value in enumerate(np.mean(error[:, horizon_step - 1, :], axis=0)):
                            camera_rows.append({"run_id": batch_dir.name, "model": model, "horizon": f"{horizon_step * interval_minutes}min", "camera_index": camera_index + 1, "mae": round(float(value), 6)})
            else:
                for camera_index, value in enumerate(np.mean(error, axis=0)):
                    camera_rows.append({"run_id": batch_dir.name, "model": model, "horizon": horizon, "camera_index": camera_index + 1, "mae": round(float(value), 6)})
    return metric_rows, camera_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """写入字典列表 CSV。"""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """生成跨批次模型评估结果。"""
    parser = argparse.ArgumentParser(description="Evaluate LibCity experiment batches for one dataset tag")
    dp.add_tag_argument(parser)
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag, interactive=True)

    experiment_root = dp.stage_dir(tag, "06_libcity_experiment")
    output_root = dp.stage_dir(tag, "08_model_evaluation")
    print(f"Dataset tag: {tag}")

    metric_rows, camera_rows = collect_results(experiment_root)
    if not metric_rows:
        raise FileNotFoundError(f"No experiment results found: {experiment_root}")
    output_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    grouped: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row.get("model", "")), str(row.get("horizon_minutes", "")))].append(row)
    summary_rows = []
    for (model, horizon), rows in sorted(grouped.items()):
        summary = {"model": model, "horizon_minutes": horizon, "experiment_count": len(rows)}
        for metric in ("MAE", "RMSE", "R2", "masked_MAE", "masked_RMSE"):
            values = [float(row[metric]) for row in rows if np.isfinite(float(row[metric]))]
            summary[f"{metric.lower()}_mean"] = round(float(np.mean(values)), 6) if values else ""
            summary[f"{metric.lower()}_std"] = round(float(np.std(values)), 6) if values else ""
        summary_rows.append(summary)
    write_csv(output_dir / "evaluation_summary.csv", summary_rows)
    write_csv(output_dir / "camera_error_summary.csv", camera_rows)
    write_csv(output_dir / "experiment_metrics.csv", metric_rows)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset_tag": tag, "source_directory": str(experiment_root), "experiment_count": len(metric_rows), "groups": len(summary_rows), "models": sorted({str(row.get("model", "")) for row in metric_rows}), "horizons_minutes": sorted({str(row.get("horizon_minutes", "")) for row in metric_rows})}
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Model evaluation saved to: {output_dir}")


if __name__ == "__main__":
    main()
