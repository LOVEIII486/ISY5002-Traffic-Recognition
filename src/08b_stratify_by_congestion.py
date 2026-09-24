"""把预测误差按**目标时刻的拥堵档位**分层，看模型在拥堵时是否更差。

为什么需要：`06` 报的是全测试集平均 MAE。如果模型在自由流时误差 3、拥堵时误差 12，
那个平均值就有误导性——模型恰恰在最需要它的场景下最不准。

标签只用于**分组**，不参与训练。这是拥堵指标唯一站得住的用法：它的定义需要同小时
全序列（含未来帧），拿它当预测目标会用到未来信息，而 `occ_px` 又与输入特征同源。
"""

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


EVAL_RATE = 0.2
HORIZON_STEPS = (1, 3, 12)
LABEL_ORDER = ("free", "moderate", "congested")


def align_to_grid(captured_at: str, interval: int) -> str:
    """把原始拍摄时间向下对齐到采样网格，与 `05` 的 `parse_time` 同一口径。"""
    parsed = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    aligned = parsed.replace(minute=(parsed.minute // interval) * interval, second=0, microsecond=0)
    return aligned.isoformat().replace("+00:00", "Z")


def read_labels(metrics_csv: Path, interval: int) -> dict[tuple[str, str], str]:
    """读 `04_traffic_metrics` 的拥堵列，键为 (camera_id, 对齐后的时间戳)。

    只保留 `measurable=True` 的行——不可测的帧没有标签，也**不应该被当成自由流**。
    同一 (相机, 时刻) 有多行时取第一条，与 `05` 的 `value_map` 口径一致。
    """
    labels: dict[tuple[str, str], str] = {}
    if not metrics_csv.is_file():
        raise SystemExit(f"拥堵指标不存在：{metrics_csv}\n先跑 04b 再跑 04。")
    with metrics_csv.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("label", "").strip() == "" or row.get("measurable") != "True":
                continue
            captured = row.get("captured_at", "")
            if not captured:
                continue
            try:
                key = (row["camera_id"], align_to_grid(captured, interval))
            except ValueError:
                continue
            labels.setdefault(key, row["label"])
    return labels


def read_timestamps(dyna_path: Path) -> list[str]:
    """读取 LibCity 数据集里的唯一时间戳（已按网格对齐）。"""
    timestamps, seen = [], set()
    with dyna_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            value = row.get("time", "")
            if value and value not in seen:
                timestamps.append(value)
                seen.add(value)
    timestamps.sort()
    return timestamps


def read_camera_ids(geo_path: Path) -> list[str]:
    """读取 geo 文件里的摄像头顺序——它必须与预测数组的最后一维一致。"""
    with geo_path.open(newline="", encoding="utf-8") as stream:
        return [row.get("geo_id", "") for row in csv.DictReader(stream)]


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取并规范化预测数组。"""
    payload = np.load(path)
    prediction = np.asarray(payload["prediction"], dtype=float)
    truth = np.asarray(payload["truth"], dtype=float)
    if prediction.ndim == 4:
        prediction = prediction[..., 0]
        truth = truth[..., 0]
    if prediction.ndim != 3:
        raise ValueError(f"Unexpected prediction shape in {path}: {prediction.shape}")
    return prediction, truth


def latest_complete_batch(experiment_root: Path) -> Path:
    """选择最新的完整实验批次。"""
    batches = sorted(experiment_root.glob("run_*"), key=lambda path: path.name)
    complete = [path for path in batches if (path / "model_comparison.csv").is_file()]
    if not complete:
        raise FileNotFoundError(f"No complete batch under {experiment_root}")
    return complete[-1]


def metrics_for(errors: np.ndarray) -> dict[str, float]:
    """按一组绝对误差算 MAE / RMSE。"""
    if errors.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "n": 0}
    return {
        "MAE": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "n": int(errors.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify forecasting error by the congestion label at the target time")
    dp.add_tag_argument(parser)
    parser.add_argument("--batch", type=Path, default=None, help="Default: latest complete batch under 06")
    parser.add_argument("--output", type=Path, default=None, help="Default: results/<tag>/08b_stratified")
    args = parser.parse_args()

    tag = dp.resolve_tag(args.dataset_tag, interactive=True)
    experiment_root = dp.stage_dir(tag, "06_libcity_experiment")
    dataset_dir = dp.stage_dir(tag, "05_libcity_dataset")
    metrics_csv = dp.stage_dir(tag, "04_traffic_metrics") / "traffic_metrics.csv"
    output_dir = args.output or dp.stage_dir(tag, "08b_stratified")
    batch = args.batch or latest_complete_batch(experiment_root)
    print(f"Dataset tag: {tag}")
    print(f"Batch: {batch.name}")

    config_path = batch / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    interval = int(config.get("interval_minutes", 5))
    input_window = int(config.get("input_window", 12))

    camera_ids = read_camera_ids(dataset_dir / "isy5002_traffic.geo")
    timestamps = read_timestamps(dataset_dir / "isy5002_traffic.dyna")
    labels = read_labels(metrics_csv, interval)
    print(f"  {len(timestamps)} 个时间点，{len(camera_ids)} 个摄像头，{len(labels)} 条可用拥堵标签")

    # 测试段起点：与 07 的 sample_labels 和 06b 的 build_samples 同一口径
    sample_total = max(0, len(timestamps) - input_window - int(config.get("output_window", 12)) + 1)
    test_start = sample_total - round(sample_total * EVAL_RATE)

    # 没有预测数组就先报这个——否则会误报成「标签对不上」，把人引到错误方向
    model_dirs = [
        path for path in sorted((batch / "models").glob("*"))
        if path.is_dir() and (path / "predictions.npz").is_file()
    ]
    if not model_dirs:
        raise SystemExit(
            f"批次 {batch.name} 里没有任何 predictions.npz，无法做分层评估。\n"
            f"  （这与拥堵标签无关——标签侧是好的。）\n"
            f"  需要先为该标签重跑：\n"
            f"    python src/06_run_libcity_experiment.py --dataset-tag {tag}"
        )

    by_label: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    per_camera: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    label_hits = 0
    label_total = 0

    for model_dir in model_dirs:
        model = model_dir.name.upper()
        prediction, truth = load_predictions(model_dir / "predictions.npz")
        for step in HORIZON_STEPS:
            index = step - 1
            if index >= prediction.shape[1]:
                continue
            horizon = step * interval
            for sample in range(prediction.shape[0]):
                target_time_index = test_start + sample + input_window + index
                if target_time_index >= len(timestamps):
                    continue
                stamp = timestamps[target_time_index]
                for col, camera_id in enumerate(camera_ids):
                    label_total += 1
                    label = labels.get((camera_id, stamp))
                    if label is None:
                        continue
                    label_hits += 1
                    error = abs(float(prediction[sample, index, col]) - float(truth[sample, index, col]))
                    by_label[(model, horizon, label)].append(error)
                    per_camera[(model, horizon, camera_id, label)].append(error)

    if not by_label:
        raise SystemExit(
            "一条标签都对不上。检查 04 是否已带拥堵列（先跑 04b 再跑 04），"
            "以及网格对齐的 interval 是否与 06 的配置一致。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (model, horizon, label), errors in sorted(by_label.items()):
        rows.append({
            "model": model, "horizon_minutes": horizon, "label": label,
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics_for(np.asarray(errors)).items()},
        })
    with (output_dir / "stratified_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    camera_rows = []
    for (model, horizon, camera_id, label), errors in sorted(per_camera.items()):
        camera_rows.append({
            "model": model, "horizon_minutes": horizon, "camera_id": camera_id, "label": label,
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics_for(np.asarray(errors)).items()},
        })
    with (output_dir / "stratified_camera_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(camera_rows[0]))
        writer.writeheader()
        writer.writerows(camera_rows)

    (output_dir / "run_metadata.json").write_text(
        json.dumps({
            "dataset_tag": tag, "batch": batch.name, "interval_minutes": interval,
            "labelled_samples": label_hits,
            "unlabelled_samples": label_total - label_hits,
            "labelled_fraction": round(label_hits / label_total, 4) if label_total else 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 结论表：同一模型同一跨度下，拥堵档相对自由流档的误差倍数
    print(f"\n{'model':7s} {'horizon':>8s} " + "".join(f"{l:>12s}" for l in LABEL_ORDER) + f"{'拥堵/自由流':>14s}")
    for model in sorted({r['model'] for r in rows}):
        for horizon in sorted({r['horizon_minutes'] for r in rows}):
            subset = {r["label"]: r for r in rows if r["model"] == model and r["horizon_minutes"] == horizon}
            cells = ""
            for label in LABEL_ORDER:
                item = subset.get(label)
                cells += f"{item['MAE']:>12.3f}" if item else f"{'-':>12s}"
            free, cong = subset.get("free"), subset.get("congested")
            ratio = f"{cong['MAE'] / free['MAE']:.2f}x" if free and cong and free["MAE"] else "-"
            print(f"{model:7s} {horizon:>6d}min {cells}{ratio:>14s}")
    print(f"\n样本标签覆盖：{label_hits}/{label_total} = {label_hits/label_total:.1%}（未覆盖的是不可测帧，已排除）")
    print(f"Saved: {output_dir / 'stratified_metrics.csv'}")


if __name__ == "__main__":
    main()
