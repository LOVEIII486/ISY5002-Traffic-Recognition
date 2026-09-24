"""Launch a Gradio web prototype for traffic detection and forecasting results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as dp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 下面这些路径都随 --dataset-tag 变化，由 main() 调用 resolve_paths() 一次填好。
# 用模块级变量是因为 Gradio 回调的签名由界面接线决定，把标签逐个透传进去很别扭。
EXPERIMENT_ROOT = Path()
METRICS_FILE = Path()
GEO_FILE = Path()
DYNA_FILE = Path()
YOLO_DETECTIONS_FILE = Path()
ANNOTATED_DIR = Path()
OUTPUT_ROOT = Path()
# CNN 权重按项目共享（不随数据集切分）
CNN_WEIGHTS = PROJECT_ROOT / "train" / "weights" / "best_lta.pt"


def resolve_paths(tag: str) -> None:
    """按数据集标签填好全部输入/输出路径。"""
    global EXPERIMENT_ROOT, METRICS_FILE, GEO_FILE, DYNA_FILE
    global YOLO_DETECTIONS_FILE, ANNOTATED_DIR, OUTPUT_ROOT
    EXPERIMENT_ROOT = dp.stage_dir(tag, "06_libcity_experiment")
    METRICS_FILE = dp.stage_dir(tag, "04_traffic_metrics") / "traffic_metrics.csv"
    GEO_FILE = dp.stage_dir(tag, "05_libcity_dataset") / "isy5002_traffic.geo"
    DYNA_FILE = dp.stage_dir(tag, "05_libcity_dataset") / "isy5002_traffic.dyna"
    YOLO_DETECTIONS_FILE = dp.stage_dir(tag, "03_yolo_detection") / "frame_detections.csv"
    ANNOTATED_DIR = dp.stage_dir(tag, "03_yolo_detection") / "annotated"
    OUTPUT_ROOT = dp.stage_dir(tag, "09_mvp_demo")
    # 拥堵列已由 04b 并入 04_traffic_metrics/traffic_metrics.csv，
    # 所以 METRICS_FILE 本身就带 occ_px / label，无需单独读 04b。


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 文件。"""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


_yolo_boxes_cache = None
_cnn = None


def read_yolo_boxes() -> dict[str, list[tuple[float, float, float, float]]]:
    """读取 YOLO 检测框，按 image_path 分组（惰性缓存）。"""
    global _yolo_boxes_cache
    if _yolo_boxes_cache is None:
        boxes: defaultdict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        for row in read_csv(YOLO_DETECTIONS_FILE):
            try:
                boxes[row["image_path"]].append((float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])))
            except (KeyError, ValueError):
                continue
        _yolo_boxes_cache = dict(boxes)
    return _yolo_boxes_cache


def load_cnn():
    """惰性加载微调后的 CNN 模型，返回 (model, device, input_size) 或 None。"""
    global _cnn
    if _cnn is None:
        if not CNN_WEIGHTS.is_file():
            return None
        sys.path.insert(0, str(PROJECT_ROOT / "train"))
        from predict import load_model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, config = load_model(CNN_WEIGHTS, device)
        _cnn = (model, device, tuple(config.get("input_size", [360, 640])))
    return _cnn


# 密度热力图可视化参数：固定色标（红 = 达到 DENSITY_MAX 的高密度），而非逐图归一化，
# 避免稀疏图里文字这类弱响应被放大成红色；低于 DENSITY_FLOOR 的密度当作背景（透明）。
DENSITY_MAX = 0.04
DENSITY_FLOOR = 0.01

_road_mask_cache: dict[str, np.ndarray | None] = {}


def load_road_mask(camera_id: str | None) -> np.ndarray | None:
    """读取该相机的道路掩码（惰性缓存）；缺失时返回 None。"""
    if not camera_id:
        return None
    if camera_id not in _road_mask_cache:
        path = dp.road_mask_path(camera_id)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        _road_mask_cache[camera_id] = None if image is None else image > 127
    return _road_mask_cache[camera_id]


def cnn_density_overlay(image_bgr, camera_id: str | None = None):
    """把 CNN 密度热力图半透明叠加到图片上（BGR 输入/输出）。

    热力图被**限制在道路掩码内**：密度 CNN 会对水面、天空和烧录文字产生虚假响应，
    而「车不在水面上」是硬几何约束。DENSITY_FLOOR 是按数值压噪声，掩码是按位置——
    两者互补，后者才是主要依据。掩码缺失时退化为不裁剪。
    """
    cnn = load_cnn()
    if cnn is None:
        return image_bgr
    from predict import predict_density
    model, device, input_size = cnn
    density = predict_density(model, image_bgr, device, input_size)
    h, w = image_bgr.shape[:2]
    density_up = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = np.clip(density_up / DENSITY_MAX, 0.0, 1.0)
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    visible = density_up >= DENSITY_FLOOR
    road = load_road_mask(camera_id)
    if road is not None:
        if road.shape != (h, w):
            road = cv2.resize(road.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        visible &= road
    mask = visible.astype(np.float32)[..., None]
    return (image_bgr * (1 - 0.6 * mask) + heatmap * (0.6 * mask)).astype(np.uint8)


def read_camera_info() -> dict[str, dict[str, str]]:
    """读取摄像头坐标和区域。"""
    cameras = {}
    for row in read_csv(GEO_FILE):
        try:
            coordinates = json.loads(row.get("coordinates", "[]"))
            cameras[row["geo_id"]] = {"camera_id": row["geo_id"], "region": row.get("region", "unknown"), "longitude": coordinates[0], "latitude": coordinates[1]}
        except (KeyError, IndexError, json.JSONDecodeError):
            continue
    return cameras


def read_dataset_timestamps() -> list[str]:
    """读取 LibCity 数据集中的唯一时间戳。"""
    dyna_file = DYNA_FILE
    timestamps = []
    seen = set()
    for row in read_csv(dyna_file):
        value = row.get("time", "")
        if value and value not in seen:
            timestamps.append(value)
            seen.add(value)
    return timestamps


def latest_batch() -> Path:
    """选择最新的完整实验批次（要求含 model_comparison.csv）。"""
    batches = sorted(EXPERIMENT_ROOT.glob("run_*"), key=lambda path: path.name)
    if not batches:
        raise FileNotFoundError(f"No experiment batch found: {EXPERIMENT_ROOT}")
    complete = [path for path in batches if (path / "model_comparison.csv").is_file()]
    if not complete:
        raise FileNotFoundError(
            f"No complete experiment batch under {EXPERIMENT_ROOT}: "
            f"none of {len(batches)} run_* directories contains model_comparison.csv. "
            "Run src/06_run_libcity_experiment.py for this dataset tag first."
        )
    return complete[-1]


def prediction_data(batch: Path, horizon: str, model: str) -> tuple[np.ndarray, np.ndarray, Path]:
    """读取指定模型的预测数组。"""
    model_dir = batch / "models" / model.lower()
    if not model_dir.is_dir():
        model_dir = batch / f"horizon_{horizon}" / model.lower()
    path = model_dir / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    payload = np.load(path)
    prediction = np.asarray(payload["prediction"], dtype=float)
    truth = np.asarray(payload["truth"], dtype=float)
    if prediction.ndim == 4:
        prediction = prediction[..., 0]
        truth = truth[..., 0]
    if prediction.ndim != 3:
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
    return prediction, truth, model_dir


def read_experiment_config(batch: Path) -> dict:
    """读取批次配置；文件缺失或损坏时返回空字典。"""
    config_path = batch / "experiment_config.json"
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


def first_horizon_minutes(config: dict) -> int:
    """最短预测档的跨度（分钟）。"""
    interval = interval_minutes_of(config)
    steps = []
    for window in config.get("output_windows", [1, 3, 12]):
        try:
            step = int(window)
        except (TypeError, ValueError):
            continue
        if step > 0:
            steps.append(step)
    return (min(steps) if steps else 1) * interval


def available_options(batch: Path) -> tuple[list[str], list[str], list[str]]:
    """读取模型、预测时长和摄像头选项。"""
    config = read_experiment_config(batch)
    interval = interval_minutes_of(config)
    output_windows = config.get("output_windows", [1, 3, 12])
    horizons = [f"{int(step) * interval}min" for step in output_windows]
    models = sorted(path.name.upper() for path in (batch / "models").iterdir() if path.is_dir()) if (batch / "models").is_dir() else sorted({path.name.upper() for horizon in batch.glob("horizon_*") for path in horizon.iterdir() if path.is_dir()})
    cameras = list(read_camera_info())
    return models, horizons, cameras


def match_tolerance_seconds(rows: list[dict]) -> float:
    """按该摄像头自身的采样间隔给时间匹配容差。"""
    stamps = []
    for row in rows:
        value = row.get("captured_at")
        if not value:
            continue
        try:
            stamps.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            continue
    stamps.sort()
    gaps = sorted((later - earlier).total_seconds() for earlier, later in zip(stamps, stamps[1:]) if later > earlier)
    if not gaps:
        return 300.0
    return max(60.0, gaps[len(gaps) // 2])


def select_metrics_row(camera_id: str, sample_index: int, timestamp: str = "") -> dict | None:
    """Pick the traffic-metrics row (carries YOLO + CNN counts) for a camera/time."""
    rows = [row for row in read_csv(METRICS_FILE) if row.get("camera_id") == str(camera_id)]
    rows.sort(key=lambda row: row.get("captured_at", ""))
    if not rows:
        return None
    selected_row = next((row for row in rows if row.get("captured_at", "").replace("+00:00", "Z") == timestamp), None)
    if selected_row is None and timestamp:
        try:
            target_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            timed_rows = [(abs(datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00")) - target_time), row) for row in rows if row.get("captured_at")]
            if timed_rows:
                distance, nearest_row = min(timed_rows, key=lambda item: item[0])
                if distance.total_seconds() <= match_tolerance_seconds(rows):
                    selected_row = nearest_row
        except (KeyError, ValueError, TypeError):
            selected_row = None
    return selected_row or rows[sample_index % len(rows)]


def image_for_camera(camera_id: str, sample_index: int, timestamp: str = "") -> Path | None:
    """按摄像头和时间戳查找示例图片。"""
    selected_row = select_metrics_row(camera_id, sample_index, timestamp)
    if selected_row is None:
        return None
    path = Path(selected_row.get("image_path", ""))
    if not path.is_file():
        return None
    annotated_dir = ANNOTATED_DIR
    annotated_name = f"{path.stem}_{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:8]}{path.suffix.lower()}"
    annotated_path = annotated_dir / annotated_name
    if annotated_path.is_file():
        return annotated_path
    fallback = next(iter(annotated_dir.glob(f"{path.stem}_*{path.suffix.lower()}")), None) if annotated_dir.is_dir() else None
    return fallback or path


def update_time_choices():
    """生成测试集内的全部采样时间点。"""
    try:
        batch = latest_batch()
        config = read_experiment_config(batch)
        model_name = available_options(batch)[0][0]
        prediction, _, _ = prediction_data(batch, f"{first_horizon_minutes(config)}min", model_name)
        timestamps = read_dataset_timestamps()
        # LibCity 的样本以 t 为输入窗口末端，第一步预测目标是 t+1。
        # 测试集从最后 round(total_samples * test_rate) 个样本开始。
        input_window = int(config.get("input_window", 12))
        output_windows = [int(step) for step in config.get("output_windows", [1, 3, 12])]
        output_window = max(output_windows) if output_windows else 12
        total_samples = max(0, len(timestamps) - input_window - output_window + 1)
        test_count = round(total_samples * 0.2)
        test_start = total_samples - test_count
        choices = []
        for index in range(prediction.shape[0]):
            timestamp_index = input_window + test_start + index
            choices.append(timestamps[timestamp_index] if timestamp_index < len(timestamps) else f"Test sample {index}")
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
    except (FileNotFoundError, ValueError):
        return gr.Dropdown(choices=[], value=None)


def show_result(camera_id: str, sample_index: str):
    """展示指定摄像头和测试时间点的三个模型结果。"""
    batch = latest_batch()
    config = read_experiment_config(batch)
    interval_minutes = interval_minutes_of(config)
    input_window = int(config.get("input_window", 12))
    horizon_minutes = first_horizon_minutes(config)
    models, _, _ = available_options(batch)
    prediction, truth, model_dir = prediction_data(batch, f"{horizon_minutes}min", models[0])
    try:
        sample_choices = update_time_choices().choices
        sample_values = [choice[1] if isinstance(choice, tuple) else choice for choice in sample_choices]
        sample = sample_values.index(sample_index) if sample_index in sample_values else int(sample_index)
    except (AttributeError, ValueError, TypeError):
        sample = 0
    sample = max(0, min(sample, prediction.shape[0] - 1))
    camera_info = read_camera_info()
    camera_ids = list(camera_info)
    camera_index = camera_ids.index(str(camera_id)) if str(camera_id) in camera_ids else 0
    camera = camera_info.get(str(camera_id), {"camera_id": camera_id, "region": "unknown", "latitude": "", "longitude": ""})
    prediction_time = sample_index[1] if isinstance(sample_index, tuple) else str(sample_index)
    metrics_row = select_metrics_row(str(camera_id), sample, prediction_time)
    yolo_count = metrics_row.get("total_vehicles", "") if metrics_row else ""
    cnn_count = metrics_row.get("cnn_count", "") if metrics_row else ""
    cong_label = metrics_row.get("label", "") if metrics_row else ""
    cong_occ = metrics_row.get("occ_px", "") if metrics_row else ""
    cong_road = metrics_row.get("road_area_frac", "") if metrics_row else ""
    result_rows = []
    for model_name in models:
        model_prediction, model_truth, _ = prediction_data(batch, f"{horizon_minutes}min", model_name)
        actual = float(model_truth[sample, 0, camera_index])
        predicted = float(model_prediction[sample, 0, camera_index])
        metric_file = model_dir.parent / model_name.lower() / "model_metrics.csv"
        metric_row = next((row for row in read_csv(metric_file) if row.get("horizon_minutes") == str(horizon_minutes)), {})
        result_rows.append([model_name, round(actual, 3), round(predicted, 3), round(abs(predicted - actual), 3), round(float(metric_row.get("MAE", "nan")), 3), round(float(metric_row.get("RMSE", "nan")), 3), round(float(metric_row.get("R2", "nan")), 3)])
    input_minutes = input_window * interval_minutes
    details = {"experiment_batch": batch.name, "prediction_time": prediction_time, "camera": camera, "forecast_target": f"selected time point ({horizon_minutes}-minute forecast)", "input_history": f"{input_window} points / {input_minutes} minutes", "detection": {"yolo_vehicles": yolo_count, "cnn_vehicles": cnn_count}, "congestion": {"label": cong_label, "occ_px": cong_occ, "road_area_frac": cong_road}, "results": result_rows}
    session_dir = OUTPUT_ROOT / "web_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / f"view_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    table = result_rows
    timestamp = prediction_time if prediction_time else "unknown"
    yolo_str = yolo_count if yolo_count != "" else "N/A"
    cnn_str = cnn_count if cnn_count != "" else "N/A"
    congestion_block = (
        "### Congestion at this time\n"
        f"- Density `occ_px`: `{cong_occ if cong_occ != '' else 'N/A'}`"
        f" (road area fraction `{cong_road if cong_road != '' else 'N/A'}`)\n"
        f"- Label: `{cong_label if cong_label else 'N/A - not measurable at this time'}`\n"
        "- Note: the label is a percentile **within this camera at this hour of day** - "
        "it says this frame is busier than this camera's own norm at this time, "
        "not an absolute service level. An empty label means the measurement was not "
        "trustworthy (typically at night) and is deliberately NOT reported as free-flow."
    )
    summary = (
        f"### Experiment details\n- Batch: `{batch.name}`\n- Prediction time: `{timestamp}`\n"
        f"- Camera: `{camera_id}`\n- Region: `{camera.get('region', 'unknown')}`\n"
        f"- Coordinates: `{camera.get('latitude', '')}, {camera.get('longitude', '')}`\n"
        f"- Input history: previous {input_window} points ({input_minutes} minutes)\n"
        f"### Detection at this time\n- YOLO detected vehicles: `{yolo_str}`\n"
        f"- CNN counted vehicles: `{cnn_str}`\n" + congestion_block
    )
    yolo_display = None
    cnn_display = None
    if metrics_row is not None:
        img = cv2.imread(metrics_row.get("image_path", ""))
        if img is not None:
            image_path_key = metrics_row.get("image_path", "")
            yolo_img = img.copy()
            for (x1, y1, x2, y2) in read_yolo_boxes().get(image_path_key, []):
                cv2.rectangle(yolo_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            yolo_display = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
            cnn_display = cv2.cvtColor(cnn_density_overlay(img, str(camera_id)), cv2.COLOR_BGR2RGB)
    return yolo_display, cnn_display, summary, table


def build_app() -> gr.Blocks:
    """构建 Gradio 页面。"""
    batch = latest_batch()
    config = read_experiment_config(batch)
    interval_minutes = interval_minutes_of(config)
    horizon_minutes = first_horizon_minutes(config)
    input_window = int(config.get("input_window", 12))
    output_window = max((int(step) for step in config.get("output_windows", [1, 3, 12])), default=12)
    models, _, cameras = available_options(batch)
    with gr.Blocks(title="ISY5002 Traffic Prediction MVP") as app:
        gr.Markdown(f"# ISY5002 Traffic Recognition and Prediction\nSelect a camera and target time to compare all trained models at the same {horizon_minutes}-minute forecast target.\n\n**Model input:** each model uses the previous {input_window} {interval_minutes}-minute vehicle-count records ({input_window * interval_minutes} minutes) for every camera. The target variable is `total_vehicles`. Models internally output {output_window} future steps; this panel compares the first ({horizon_minutes}-minute) step at one target time. Longer-horizon metrics remain available in the evaluation files.")
        with gr.Row():
            camera = gr.Dropdown(cameras, value=cameras[0] if cameras else None, label="Camera ID")
            sample = gr.Dropdown([], label="Test time point")
        run_button = gr.Button("Show result", variant="primary")
        with gr.Row():
            yolo_image = gr.Image(label="YOLO detection (green boxes)", type="numpy")
            cnn_image = gr.Image(label="CNN density heatmap (red = dense)", type="numpy")
        details = gr.Markdown()
        table = gr.Dataframe(headers=["Model", "Actual", "Predicted", "Absolute error", "Test MAE", "Test RMSE", "Test R2"], datatype=["str", "number", "number", "number", "number", "number", "number"], label="Traffic predictions at selected time")
        run_button.click(show_result, inputs=[camera, sample], outputs=[yolo_image, cnn_image, details, table])
        app.load(update_time_choices, inputs=[], outputs=sample)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio traffic prediction MVP for one dataset tag")
    dp.add_tag_argument(parser)
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag, interactive=True)
    resolve_paths(tag)
    print(f"Dataset tag: {tag}")
    print(f"Experiment batch: {latest_batch()}")
    print("Starting Gradio traffic prediction MVP")
    build_app().launch(allowed_paths=[str(PROJECT_ROOT)])


if __name__ == "__main__":
    main()
