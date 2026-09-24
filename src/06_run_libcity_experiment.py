"""Run one LibCity traffic prediction experiment on the prepared YOLO data."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as dp
import numpy as np
import torch


# 实验模型；LibCity 会把 GRU 映射到 traffic_speed_prediction.RNN
MODELS = ("FNN", "GRU", "STGCN")
DATASET_NAME = "isy5002_traffic"
INPUT_WINDOW = 12
OUTPUT_WINDOW = 1
# 兜底采样间隔；实际值由 prepare_inputs() 从 metrics 的 captured_at 推断
# （两批数据分别是 5 分钟和 10 分钟，写死会让时间轴和预测跨度都错位）。
DEFAULT_INTERVAL_MINUTES = 5
MAX_EPOCH = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBCITY_ROOT = PROJECT_ROOT / "third_party" / "LibCity"
# LibCity 的 raw_data 目录保持共享：prepare_inputs() 每次运行开头都会重写它，
# 顺序执行不会串味。代价是两批数据不能并发跑（并发会抢同一个目录）。
LIBCITY_RAW = LIBCITY_ROOT / "raw_data" / DATASET_NAME


def prepare_inputs(metrics_file: Path, dataset_dir: Path, output_window: int = OUTPUT_WINDOW, interval_minutes: int | None = None) -> dict:
    """检查并生成 LibCity 所需的时间序列数据。"""
    if not metrics_file.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    # 复用前一阶段脚本，确保检测结果被转换
    import importlib.util
    converter_spec = importlib.util.spec_from_file_location("dataset_converter", PROJECT_ROOT / "src" / "05_prepare_libcity_dataset.py")
    converter = importlib.util.module_from_spec(converter_spec)
    assert converter_spec and converter_spec.loader
    converter_spec.loader.exec_module(converter)
    rows = converter.read_metrics(metrics_file)
    metadata = converter.load_camera_metadata(converter.DEFAULT_CAMERA_METADATA)
    if interval_minutes is None:
        interval_minutes = converter.detect_interval_minutes(rows, DEFAULT_INTERVAL_MINUTES)
    print(f"Sampling interval: {interval_minutes} minutes")
    dataset_metadata = converter.write_atomic_files(rows, metadata, dataset_dir, interval_minutes)

    LIBCITY_RAW.mkdir(parents=True, exist_ok=True)
    for suffix in (".geo", ".rel", ".dyna"):
        shutil.copy2(dataset_dir / f"isy5002_traffic{suffix}", LIBCITY_RAW / f"isy5002_traffic{suffix}")
    config = {
        "info": {
            "dataset": DATASET_NAME,
            "data_col": ["total_vehicles"],
            "load_external": True,
            "add_time_in_day": True,
            "add_day_in_week": True,
            "train_rate": 0.6,
            "eval_rate": 0.2,
            "input_window": INPUT_WINDOW,
            "output_window": output_window,
            "time_interval": interval_minutes * 60,
        }
    }
    (LIBCITY_RAW / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return dataset_metadata


def run_experiment(model_name: str, exp_id: int, output_dir: Path, metadata: dict, output_window: int = 12, interval_minutes: int = DEFAULT_INTERVAL_MINUTES, dataset_tag: str = "") -> list[dict]:
    """调用 LibCity 官方 pipeline 并保存结果。"""
    if not LIBCITY_ROOT.is_dir():
        raise FileNotFoundError(f"LibCity submodule not found: {LIBCITY_ROOT}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset_tag": dataset_tag,
        "exp_id": exp_id,
        "model": model_name,
        "dataset": DATASET_NAME,
        "input_window": INPUT_WINDOW,
        "output_window": output_window,
        "interval_minutes": interval_minutes,
        "train_rate": 0.6,
        "validation_rate": 0.2,
        "test_rate": 0.2,
        "max_epoch": MAX_EPOCH,
        "device": "auto_cpu_or_gpu",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # LibCity 只按 exp_id 缓存，而评估产物文件名只含数据集名（各数据集相同）。
    # 记下评估前的文件集合，评估后必须出现**新**文件；否则说明读到的是别的数据集
    # 落在同一 exp_id 缓存里的旧结果——那种情况会静默产出看似正常的指标。
    evaluator_name = "RNN" if model_name == "GRU" else model_name
    evaluation_dir = LIBCITY_ROOT / "libcity" / "cache" / str(exp_id) / "evaluate_cache"
    evaluation_pattern = f"*_{evaluator_name}_isy5002_traffic.csv"
    stale_evaluations = {path.name for path in evaluation_dir.glob(evaluation_pattern)} if evaluation_dir.is_dir() else set()

    # LibCity 使用相对路径读取 libcity/config 和 raw_data
    original_directory = Path.cwd()
    sys.path.insert(0, str(LIBCITY_ROOT))
    try:
        import os
        os.chdir(LIBCITY_ROOT)
        # 兼容 LibCity 旧代码与新版 NumPy
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]
        # 训练执行器仅使用 Ray 的日志与检查点接口；无调参时提供轻量兼容实现
        if "ray" not in sys.modules:
            @contextmanager
            def checkpoint_dir(step=0):
                yield str(LIBCITY_ROOT / "libcity" / "cache" / str(exp_id) / "model_cache")
            ray_tune = types.SimpleNamespace(checkpoint_dir=checkpoint_dir, report=lambda **kwargs: None)
            sys.modules["ray"] = types.ModuleType("ray")
            sys.modules["ray"].tune = ray_tune
            sys.modules["ray.tune"] = ray_tune
        if "tensorboard" not in sys.modules:
            class SummaryWriter:
                def __init__(self, *args, **kwargs):
                    pass
                def add_scalar(self, *args, **kwargs):
                    pass
                def close(self):
                    pass
            tensorboard = types.ModuleType("tensorboard")
            tensorboard.SummaryWriter = SummaryWriter
            sys.modules["tensorboard"] = tensorboard
            torch_tensorboard = types.ModuleType("torch.utils.tensorboard")
            torch_tensorboard.SummaryWriter = SummaryWriter
            sys.modules["torch.utils.tensorboard"] = torch_tensorboard
        from libcity.config import ConfigParser
        from libcity.data import get_dataset
        from libcity.utils import set_random_seed
        from libcity.model import loss as libcity_loss
        original_r2 = libcity_loss.r2_score_torch
        libcity_loss.r2_score_torch = lambda preds, labels: torch.as_tensor(original_r2(preds, labels))
        original_evar = libcity_loss.explained_variance_score_torch
        libcity_loss.explained_variance_score_torch = lambda preds, labels: torch.as_tensor(original_evar(preds, labels))
        import importlib.util as import_util
        model_file = "RNN.py" if model_name == "GRU" else f"{model_name}.py"
        model_spec = import_util.spec_from_file_location(f"isy5002_libcity_{model_name.lower()}", LIBCITY_ROOT / "libcity" / "model" / "traffic_speed_prediction" / model_file)
        model_module = import_util.module_from_spec(model_spec)
        assert model_spec and model_spec.loader
        model_spec.loader.exec_module(model_module)
        executor_spec = import_util.spec_from_file_location("isy5002_libcity_executor", LIBCITY_ROOT / "libcity" / "executor" / "traffic_state_executor.py")
        executor_module = import_util.module_from_spec(executor_spec)
        assert executor_spec and executor_spec.loader
        # 绕过 LibCity executor 包中与本实验无关的可选依赖
        executor_package = types.ModuleType("libcity.executor")
        executor_package.__path__ = [str(LIBCITY_ROOT / "libcity" / "executor")]
        sys.modules["libcity.executor"] = executor_package
        abstract_spec = import_util.spec_from_file_location("libcity.executor.abstract_executor", LIBCITY_ROOT / "libcity" / "executor" / "abstract_executor.py")
        abstract_module = import_util.module_from_spec(abstract_spec)
        assert abstract_spec and abstract_spec.loader
        abstract_spec.loader.exec_module(abstract_module)
        sys.modules["libcity.executor.abstract_executor"] = abstract_module
        executor_spec.loader.exec_module(executor_module)

        other_args = {
            "gpu": False,
            "max_epoch": MAX_EPOCH,
            "batch_size": 32,
            "input_window": INPUT_WINDOW,
            "output_window": output_window,
            "train_rate": 0.6,
            "eval_rate": 0.2,
            "data_col": ["total_vehicles"],
            "load_external": True,
            "add_time_in_day": True,
            "add_day_in_week": True,
            "pad_with_last_sample": False,
            "cache_dataset": False,
            "exp_id": exp_id,
        }
        config = ConfigParser("traffic_state_pred", model_name, DATASET_NAME, saved_model=True, train=True, other_args=other_args)
        set_random_seed(config.get("seed", 0))
        dataset = get_dataset(config)
        train_data, valid_data, test_data = dataset.get_data()
        data_feature = dataset.get_data_feature()
        model = model_module.FNN(config, data_feature) if model_name == "FNN" else model_module.RNN(config, data_feature) if model_name == "GRU" else model_module.STGCN(config, data_feature)
        executor = executor_module.TrafficStateExecutor(config, model, data_feature)
        executor.train(train_data, valid_data)
        # train() 返回前会自动加载验证集最优 epoch；保存该状态供 MVP 推理使用
        executor.save_model(str(output_dir / "best_model.tar"))
        result = {"test_result": executor.evaluate(test_data)}
    finally:
        os.chdir(original_directory)

    # 执行器把逐步指标写进 LibCity cache；必须是本次新写出的文件
    evaluation_files = sorted(evaluation_dir.glob(evaluation_pattern))
    if not evaluation_files or evaluation_files[-1].name in stale_evaluations:
        raise RuntimeError(
            f"LibCity exp_id {exp_id} produced no new evaluation file in {evaluation_dir} "
            f"(pattern {evaluation_pattern}); refusing to reuse a stale result. "
            "Check that exp_base is unique per dataset tag."
        )

    test_result = result.get("test_result", {}) if isinstance(result, dict) else {}
    metrics = {str(key): _to_scalar(value) for key, value in test_result.items()} if isinstance(test_result, dict) else {}
    prediction_files = sorted(evaluation_dir.glob(f"*_{evaluator_name}_isy5002_traffic_predictions.npz"))
    if prediction_files:
        shutil.copy2(prediction_files[-1], output_dir / "predictions.npz")
    if evaluation_files:
        shutil.copy2(evaluation_files[-1], output_dir / "evaluation_details.csv")
    if evaluation_files:
        with evaluation_files[-1].open(newline="", encoding="utf-8") as stream:
            evaluation_rows = list(csv.DictReader(stream))
        if evaluation_rows:
            metrics.update({key: _to_scalar(value) for key, value in evaluation_rows[-1].items()})
    prediction_array = np.load(prediction_files[-1]) if prediction_files else None
    all_predictions = prediction_array["prediction"] if prediction_array is not None else None
    all_truth = prediction_array["truth"] if prediction_array is not None else None
    horizon_rows = []
    for horizon_step in (1, 3, 12):
        step_index = horizon_step - 1
        row = dict(metrics)
        if all_predictions is not None and all_truth is not None and all_predictions.shape[1] >= horizon_step:
            predicted_step = np.asarray(all_predictions[:, step_index, ..., 0], dtype=float)
            truth_step = np.asarray(all_truth[:, step_index, ..., 0], dtype=float)
            difference = predicted_step - truth_step
            valid_mask = np.abs(truth_step) > 1e-6
            valid_errors = difference[valid_mask]
            row.update({
                "MAE": float(np.mean(np.abs(difference))),
                "MSE": float(np.mean(difference ** 2)),
                "RMSE": float(np.sqrt(np.mean(difference ** 2))),
                "R2": float(1 - np.sum(difference ** 2) / max(np.sum((truth_step - np.mean(truth_step)) ** 2), 1e-12)),
                "masked_MAE": float(np.mean(np.abs(valid_errors))) if valid_errors.size else 0.0,
                "masked_MSE": float(np.mean(valid_errors ** 2)) if valid_errors.size else 0.0,
                "masked_RMSE": float(np.sqrt(np.mean(valid_errors ** 2))) if valid_errors.size else 0.0,
            })
        row.update({
            "model": model_name,
            "camera_count": metadata.get("node_count", 0),
            "horizon_minutes": horizon_step * interval_minutes,
            "experiment_id": exp_id,
            "prediction_file": "predictions.npz" if prediction_files else "",
            "model_weights": "best_model.tar" if (output_dir / "best_model.tar").is_file() else "",
        })
        horizon_rows.append(row)
    with (output_dir / "model_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(horizon_rows[0]))
        writer.writeheader()
        writer.writerows(horizon_rows)
    (output_dir / "libcity_result.json").write_text(json.dumps(result, default=_to_scalar, indent=2), encoding="utf-8")
    return horizon_rows


def _to_scalar(value):
    """将 numpy 或 torch 标量转换为 JSON 基础类型。"""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LibCity traffic prediction experiments for one dataset tag")
    dp.add_tag_argument(parser)
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag)

    metrics_file = dp.stage_dir(tag, "04_traffic_metrics") / "traffic_metrics.csv"
    dataset_dir = dp.stage_dir(tag, "05_libcity_dataset")
    output_dir = dp.stage_dir(tag, "06_libcity_experiment")
    exp_base = dp.exp_base(tag)
    print(f"Dataset tag: {tag} (LibCity exp_id base {exp_base})")

    print("Starting LibCity model comparison")
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    batch_dir = output_dir / run_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    scenarios = (1, 3, 12)
    dataset_metadata = prepare_inputs(metrics_file, dataset_dir, 12)
    interval_minutes = int(dataset_metadata["interval_minutes"])
    all_metrics = []
    batch_config = {"run_id": run_id, "dataset_tag": tag, "models": MODELS, "output_windows": scenarios, "input_window": INPUT_WINDOW, "interval_minutes": interval_minutes, "exp_base": exp_base, "features": ["total_vehicles", "time_in_day", "day_in_week"], "max_epoch": MAX_EPOCH, "created_at": datetime.now(timezone.utc).isoformat()}
    (batch_dir / "experiment_config.json").write_text(json.dumps(batch_config, indent=2), encoding="utf-8")
    for model_index, model_name in enumerate(MODELS, start=1):
        print(f"Starting {model_name} experiment (multi-horizon output)")
        model_metrics = run_experiment(model_name, exp_base + model_index, batch_dir / "models" / model_name.lower(), dataset_metadata, 12, interval_minutes, tag)
        for metric_row in model_metrics:
            metric_row["run_id"] = run_id
            all_metrics.append(metric_row)
        print(f"Completed {model_name} experiment")
    comparison_file = batch_dir / "model_comparison.csv"
    fields = sorted({key for row in all_metrics for key in row})
    with comparison_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_metrics)
    history_file = output_dir / "all_experiments.csv"
    previous_rows = []
    if history_file.is_file():
        with history_file.open(newline="", encoding="utf-8-sig") as stream:
            previous_rows = list(csv.DictReader(stream))
    history_fields = list(dict.fromkeys(key for row in [*previous_rows, *all_metrics] for key in row))
    with history_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history_fields)
        writer.writeheader()
        writer.writerows(previous_rows)
        writer.writerows(all_metrics)
    print(f"Model comparison saved to: {comparison_file}")


if __name__ == "__main__":
    main()
