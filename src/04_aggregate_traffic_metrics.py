"""Aggregate YOLO detections into one camera-level traffic table."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as dp


VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle")
TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}Z)")


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 文件。"""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_cnn_counts(path: Path) -> dict[tuple[str, str], float]:
    """读取 CNN 计数，键为 (camera_id, image_path)。"""
    counts: dict[tuple[str, str], float] = {}
    if not path.is_file():
        return counts
    for row in read_csv(path):
        key = (row.get("camera_id", ""), row.get("image_path", ""))
        try:
            counts[key] = float(row["cnn_count"])
        except (TypeError, ValueError):
            continue
    return counts


def infer_capture_time(image_path: str) -> str:
    """从旧版文件名解析时间，无法解析时使用文件修改时间。"""
    match = TIMESTAMP_RE.search(Path(image_path).name)
    if match:
        return datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
    try:
        return datetime.fromtimestamp(Path(image_path).stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def read_congestion(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """读取 04b 的拥堵指标，键为 (camera_id, 图片文件名)——按文件名连接，因旧标签存的是死路径。"""
    rows: dict[tuple[str, str], dict[str, str]] = {}
    if not path.is_file():
        return rows
    for row in read_csv(path):
        key = (row.get("camera_id", ""), Path(row.get("image_path", "")).name)
        rows[key] = row
    return rows


CONGESTION_FIELDS = ("occ_px", "occ_gnd", "road_area_frac", "residual_occupancy", "pct_cam_hour", "label", "measurable")


def aggregate(
    detections: list[dict[str, str]],
    manifest: list[dict[str, str]],
    cnn_counts: dict[tuple[str, str], float] | None = None,
    congestion: dict[tuple[str, str], dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    """按摄像头和图片聚合检测结果，并合并 CNN 计数与 `04b` 的密度/拥堵标签。"""
    grouped: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for detection in detections:
        grouped[(detection.get("camera_id", ""), detection.get("image_path", ""))].append(detection)
    for item in manifest:
        grouped.setdefault((item.get("camera_id", ""), item.get("image_path", "")), [])
    cnn_counts = cnn_counts or {}
    congestion = congestion or {}

    rows = []
    for (camera_id, image_path), items in sorted(grouped.items()):
        confidences = [float(item["confidence"]) for item in items if item.get("confidence")]
        row: dict[str, object] = {
            "camera_id": camera_id,
            "captured_at": infer_capture_time(image_path),
            "total_vehicles": len(items),
            "average_confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0,
            "max_confidence": round(max(confidences), 6) if confidences else 0.0,
            "image_path": image_path,
        }
        for class_name in VEHICLE_CLASSES:
            row[f"{class_name}_count"] = sum(item.get("class_name", "").lower() == class_name for item in items)
        cnn = cnn_counts.get((camera_id, image_path))
        if cnn is not None:
            row["cnn_count"] = round(cnn, 2)
            row["count_diff"] = round(cnn - len(items), 2)
            row["count_abs_error"] = round(abs(cnn - len(items)), 2)
        else:
            row["cnn_count"] = ""
            row["count_diff"] = ""
            row["count_abs_error"] = ""
        congestion_row = congestion.get((camera_id, Path(image_path).name))
        for field_name in CONGESTION_FIELDS:
            row[field_name] = congestion_row.get(field_name, "") if congestion_row else ""
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate YOLO detections into camera-level traffic metrics")
    dp.add_tag_argument(parser)
    parser.add_argument("--detections", type=Path, default=None, help="YOLO detection CSV (default: results/<tag>/03_yolo_detection/frame_detections.csv)")
    parser.add_argument("--manifest", type=Path, default=None, help="Image manifest CSV (default: results/<tag>/02_dataset/image_manifest.csv)")
    parser.add_argument("--output", type=Path, default=None, help="Metrics output directory (default: results/<tag>/04_traffic_metrics)")
    parser.add_argument("--cnn-counts", type=Path, default=None, help="CNN counts CSV (optional; default: results/<tag>/03b_cnn_count/cnn_counts.csv)")
    parser.add_argument("--congestion", type=Path, default=None, help="04b congestion CSV (optional; default: results/<tag>/04b_congestion/congestion_metrics.csv)")
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag)
    detections_path = args.detections if args.detections is not None else dp.stage_dir(tag, "03_yolo_detection") / "frame_detections.csv"
    manifest_path = args.manifest if args.manifest is not None else dp.stage_dir(tag, "02_dataset") / "image_manifest.csv"
    output = args.output if args.output is not None else dp.stage_dir(tag, "04_traffic_metrics")
    cnn_counts_path = args.cnn_counts if args.cnn_counts is not None else dp.stage_dir(tag, "03b_cnn_count") / "cnn_counts.csv"
    congestion_path = args.congestion if args.congestion is not None else dp.stage_dir(tag, "04b_congestion") / "congestion_metrics.csv"
    print(f"Dataset tag: {tag}")
    detections = read_csv(detections_path)
    manifest = read_csv(manifest_path) if manifest_path.is_file() else []
    cnn_counts = read_cnn_counts(cnn_counts_path)
    congestion = read_congestion(congestion_path)
    if not congestion:
        print(f"提示：没有 04b 拥堵指标（{congestion_path.name}），occ_px/label 列将为空白")
    rows = aggregate(detections, manifest, cnn_counts, congestion)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["camera_id", "captured_at", "total_vehicles", "car_count", "motorcycle_count", "bus_count", "truck_count", "bicycle_count", "average_confidence", "max_confidence", "image_path", "cnn_count", "count_diff", "count_abs_error", *CONGESTION_FIELDS]
    with (output / "traffic_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated {sum(int(row['total_vehicles']) for row in rows)} vehicles into {len(rows)} rows")
    print(f"Saved metrics to: {output / 'traffic_metrics.csv'}")


if __name__ == "__main__":
    main()
