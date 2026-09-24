"""Convert camera-level traffic metrics into LibCity atomic files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as dp


# 相机参考元数据是共享的（属于相机而非数据集）。06 通过 importlib 读取这个名字，
# 因此它必须保持模块级常量、且不掺数据集标签。
DEFAULT_CAMERA_METADATA = dp.shared_dir("00_lta_traffic_images") / "camera_locations.json"
VEHICLE_FIELDS = ("total_vehicles", "car_count", "motorcycle_count", "bus_count", "truck_count", "bicycle_count")
# 区域分组同时覆盖两批数据：2025-10 那批含 2706/4707，2026-09 那批含 4798/4799。
# 只有同一区域的摄像头之间才会生成 rel 边，所以缺失的摄像头会被标成 unknown 并孤立。
REGION_CAMERAS = {
    "Causeway": {"2701", "2702", "2704", "2706"},
    "Second_Link": {"4703", "4707", "4712", "4713"},
    "Sentosa_Gateway": {"4798", "4799"},
}
REFERENCE_COORDINATES = {
    "2701": (1.447023728, 103.7716543),
    "2702": (1.445554109, 103.7683397),
    "2704": (1.429588536, 103.769311),
    "2706": (1.414142, 103.771168),
    "4703": (1.348697862, 103.6350413),
    "4707": (1.33344648135658, 103.652700847056),
    "4712": (1.341244001, 103.6439134),
    "4713": (1.347645829, 103.6366955),
    "4798": (1.25999999687243, 103.823611110166),
    "4799": (1.26027777363278, 103.823888890049),
}


def read_metrics(path: Path) -> list[dict[str, str]]:
    """读取摄像头级交通指标。"""
    if not path.is_file():
        raise FileNotFoundError(f"Metrics CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"camera_id", "captured_at", "total_vehicles"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Metrics CSV is missing columns: {sorted(missing)}")
    return rows


def detect_interval_minutes(rows: list[dict[str, str]], default: int = 5) -> int:
    """从 captured_at 的间隔中位数推断采样间隔（两批数据分别是 5 / 10 分钟）。"""
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
    gaps = sorted(
        (later - earlier).total_seconds() / 60.0
        for earlier, later in zip(stamps, stamps[1:])
        if later > earlier
    )
    if not gaps:
        return default
    return max(1, int(round(gaps[len(gaps) // 2])))


def parse_time(value: str, interval: int) -> datetime:
    """解析 ISO 时间并对齐到固定分钟间隔。"""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    parsed = parsed.astimezone(timezone.utc)
    minute = (parsed.minute // interval) * interval
    return parsed.replace(minute=minute, second=0, microsecond=0)


def region_for(camera_id: str) -> str:
    """根据已知摄像头组提供交通走廊元数据。"""
    for region, cameras in REGION_CAMERAS.items():
        if camera_id in cameras:
            return region
    return "unknown"


def load_camera_metadata(path: Path) -> dict[str, dict]:
    """读取摄像头位置 JSON；旧版摄像头缺少位置时保留占位坐标。"""
    metadata = {
        camera_id: {"Latitude": latitude, "Longitude": longitude}
        for camera_id, (latitude, longitude) in REFERENCE_COORDINATES.items()
    }
    if not path.is_file():
        return metadata
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Camera metadata is not valid JSON: {path}") from exc
    for item in payload.get("cameras", []):
        if item.get("CameraID") is not None:
            metadata[str(item["CameraID"])] = item
    return metadata


def write_atomic_files(
    rows: list[dict[str, str]],
    metadata: dict[str, dict],
    output_dir: Path,
    interval: int,
    dataset_tag: str = "",
    camera_metadata_source: Path | None = None,
) -> dict[str, object]:
    """写入 geo、rel 和 dyna 文件。"""
    camera_ids = sorted({str(row["camera_id"]) for row in rows})
    camera_index = {camera_id: index for index, camera_id in enumerate(camera_ids)}
    parsed_rows = []
    for row in rows:
        try:
            parsed_rows.append((parse_time(row["captured_at"], interval), str(row["camera_id"]), row))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid captured_at for camera {row.get('camera_id')}: {row.get('captured_at')}") from exc
    if not parsed_rows:
        raise ValueError("Metrics CSV contains no valid rows")

    first_time = min(item[0] for item in parsed_rows)
    last_time = max(item[0] for item in parsed_rows)
    value_map: dict[tuple[datetime, str], dict[str, str]] = {}
    for timestamp, camera_id, row in parsed_rows:
        value_map.setdefault((timestamp, camera_id), row)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "isy5002_traffic"
    with prefix.with_suffix(".geo").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["geo_id", "type", "coordinates", "region"])
        for camera_id in camera_ids:
            item = metadata.get(camera_id, {})
            latitude = item.get("Latitude")
            longitude = item.get("Longitude")
            coordinates = [float(longitude), float(latitude)] if latitude is not None and longitude is not None else [0.0, 0.0]
            writer.writerow([camera_id, "Point", json.dumps(coordinates), region_for(camera_id)])

    with prefix.with_suffix(".rel").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rel_id", "type", "origin_id", "destination_id"])
        rel_id = 0
        for origin in camera_ids:
            for destination in camera_ids:
                if origin != destination and region_for(origin) == region_for(destination):
                    writer.writerow([rel_id, "geo", origin, destination])
                    rel_id += 1

    with prefix.with_suffix(".dyna").open("w", newline="", encoding="utf-8") as stream:
        fields = ["dyna_id", "type", "time", "entity_id", *VEHICLE_FIELDS, "is_imputed"]
        writer = csv.writer(stream)
        writer.writerow(fields)
        dyna_id = 0
        timestamp_values = list(_time_range(first_time, last_time, interval))
        for camera_id in camera_ids:
            for timestamp in timestamp_values:
                source = value_map.get((timestamp, camera_id))
                is_imputed = 0 if source else 1
                values = [source.get(field, "0") if source else "0" for field in VEHICLE_FIELDS]
                writer.writerow([dyna_id, "state", timestamp.isoformat().replace("+00:00", "Z"), camera_id, *values, is_imputed])
                dyna_id += 1

    metadata_output = {
        "dataset_name": "isy5002_traffic",
        "dataset_tag": dataset_tag,
        "camera_metadata_source": str(camera_metadata_source) if camera_metadata_source else "",
        "node_count": len(camera_ids),
        "camera_ids": camera_ids,
        "start_time": first_time.isoformat().replace("+00:00", "Z"),
        "end_time": last_time.isoformat().replace("+00:00", "Z"),
        "interval_minutes": interval,
        "imputed_records": sum(1 for timestamp in _time_range(first_time, last_time, interval) for camera_id in camera_ids if (timestamp, camera_id) not in value_map),
        "note": "Coordinates are [longitude, latitude]. Missing camera coordinates use [0.0, 0.0].",
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata_output, indent=2), encoding="utf-8")
    return metadata_output


def _time_range(start: datetime, end: datetime, interval: int):
    """生成固定间隔时间点。"""
    current = start
    while current <= end:
        yield current
        current += timedelta(minutes=interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert traffic metrics to LibCity atomic files")
    dp.add_tag_argument(parser)
    parser.add_argument("--metrics", type=Path, default=None, help="Camera-level traffic metrics CSV (default: results/<tag>/04_traffic_metrics/traffic_metrics.csv)")
    parser.add_argument("--camera-metadata", type=Path, default=DEFAULT_CAMERA_METADATA, help="Camera locations JSON (shared across datasets)")
    parser.add_argument("--output", type=Path, default=None, help="LibCity dataset output directory (default: results/<tag>/05_libcity_dataset)")
    parser.add_argument("--interval-minutes", type=int, default=None, help="Target time interval in minutes; omit to infer it from captured_at")
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag)
    metrics_path = args.metrics if args.metrics is not None else dp.stage_dir(tag, "04_traffic_metrics") / "traffic_metrics.csv"
    output = args.output if args.output is not None else dp.stage_dir(tag, "05_libcity_dataset")
    print(f"Dataset tag: {tag}")
    if args.interval_minutes is not None and args.interval_minutes <= 0:
        raise ValueError("interval-minutes must be greater than zero")
    rows = read_metrics(metrics_path)
    interval = args.interval_minutes if args.interval_minutes is not None else detect_interval_minutes(rows)
    result = write_atomic_files(
        rows,
        load_camera_metadata(args.camera_metadata),
        output,
        interval,
        dataset_tag=tag,
        camera_metadata_source=args.camera_metadata,
    )
    print(f"Prepared LibCity dataset with {result['node_count']} cameras")
    print(f"Time interval: {interval} minutes" + ("" if args.interval_minutes is not None else " (inferred from captured_at)"))
    print(f"Generated files in: {output}")
    if result["imputed_records"]:
        print(f"Imputed missing camera-time records: {result['imputed_records']}")


if __name__ == "__main__":
    main()
