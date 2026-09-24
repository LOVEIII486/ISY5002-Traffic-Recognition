"""把车辆计数换算成道路密度，并按同相机同时段的百分位给出拥堵档位。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as dp
import road_geometry as rg

TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}Z)")

# 可测性门控阈值
MIN_MEDIAN_CONFIDENCE = 0.35   # 夜间实测 0.10–0.29，白天 0.44–0.55
MIN_MEDIAN_BOX_HEIGHT = 10.0   # 远车最小约 6–10 px，低于 10 px 基本不可信
MIN_BUCKET_N = 8               # 百分位桶的最小样本数，低于则回退
LABEL_FREE = "free"
LABEL_MODERATE = "moderate"
LABEL_CONGESTED = "congested"


def parse_capture_time(image_path: str) -> datetime | None:
    """从文件名解析拍摄时间（UTC）。"""
    match = TIMESTAMP_RE.search(Path(image_path).name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def sgt_hour(captured: datetime | None) -> int | None:
    """SGT 小时（UTC+8）。"""
    if captured is None:
        return None
    return (captured.hour + 8) % 24


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_cnn_counts(path: Path) -> dict[tuple[str, str], float]:
    """读取 CNN 计数，键为 (camera_id, image_path)。"""
    counts: dict[tuple[str, str], float] = {}
    if not path.is_file():
        return counts
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                counts[(row.get("camera_id", ""), row.get("image_path", ""))] = float(row["cnn_count"])
            except (KeyError, TypeError, ValueError):
                continue
    return counts


def read_cnn_centers(path: Path) -> dict[str, list[tuple[float, float]]]:
    """读取 CNN 密度中心点，键为 image_path，坐标已是原图像素。"""
    centers: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if not path.is_file():
        return centers
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                centers[row["image_path"]].append((float(row["x"]), float(row["y"])))
            except (KeyError, TypeError, ValueError):
                continue
    return centers


def load_mask(camera_id: str) -> np.ndarray | None:
    """读取二值道路掩码；缺失时返回 None。"""
    path = dp.road_mask_path(camera_id)
    if not path.is_file():
        return None
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("OpenCV is required. Install it with: pip install opencv-python") from exc
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else image > 127


def points_in_mask(mask: np.ndarray, points: list[tuple[float, float]]) -> int:
    """统计落在掩码内的点数。"""
    if not points:
        return 0
    h, w = mask.shape[:2]
    inside = 0
    for x, y in points:
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi]:
            inside += 1
    return inside


def build_clean_plate(camera_id: str, image_paths: list[Path], max_frames: int = 150):
    """时间中值底图：多帧逐像素取中值，运动车辆被抹掉。"""
    import cv2

    if len(image_paths) < 5:
        return None
    step = max(1, len(image_paths) // max_frames)
    samples = []
    for path in image_paths[::step][:max_frames]:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        samples.append(cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA))
    if len(samples) < 5:
        return None
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def residual_occupancy(gray_small: np.ndarray, plate: np.ndarray, mask_small: np.ndarray, tau: int = 25) -> float:
    """掩码内「与底图明显不同」的像素比例（不依赖检测器，夜间可用）。"""
    if plate is None or gray_small.shape != plate.shape:
        return float("nan")
    changed = np.abs(gray_small.astype(np.int16) - plate.astype(np.int16)) > tau
    denominator = int(mask_small.sum())
    return float((changed & mask_small).sum() / denominator) if denominator else float("nan")


def percentile_rank(sorted_values: list[float], value: float) -> float:
    """value 在有序列表中的百分位（0-100）。"""
    if not sorted_values:
        return 0.0
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return 100.0 * lo / len(sorted_values)


def assign_labels(rows: list[dict], bucket: str) -> dict:
    """按 (相机, 时段) 百分位给每帧打档位标签，桶太薄时逐级回退。"""
    def key_of(row, level):
        if level == "camera_hour":
            return (row["camera_id"], row["hour_sgt"])
        if level == "camera":
            return (row["camera_id"],)
        return ("__global__",)

    levels = ["camera_hour", "camera", "global"] if bucket == "camera_hour" else ["global"]
    groups: dict[str, dict] = {}
    for level in levels:
        buckets: dict[tuple, list[float]] = defaultdict(list)
        for row in rows:
            if row["measurable"] and row["occ_px"] is not None:
                buckets[key_of(row, level)].append(row["occ_px"])
        groups[level] = {
            k: {"values": sorted(v), "p50": float(np.percentile(v, 50)), "p85": float(np.percentile(v, 85)), "n": len(v)}
            for k, v in buckets.items()
        }

    thresholds = {}
    for row in rows:
        if not row["measurable"] or row["occ_px"] is None:
            row["label"] = ""
            row["label_source"] = "not_measurable"
            row["pct_cam_hour"] = None
            continue
        for level in levels:
            group = groups[level].get(key_of(row, level))
            if group and group["n"] >= MIN_BUCKET_N:
                row["pct_cam_hour"] = round(percentile_rank(group["values"], row["occ_px"]), 1)
                row["label"] = (
                    LABEL_CONGESTED if row["occ_px"] > group["p85"]
                    else LABEL_FREE if row["occ_px"] <= group["p50"]
                    else LABEL_MODERATE
                )
                row["label_source"] = level
                thresholds[f"{level}:{key_of(row, level)}"] = {
                    "p50": round(group["p50"], 5), "p85": round(group["p85"], 5), "n": group["n"],
                }
                break
        else:
            row["label"] = ""
            row["label_source"] = "no_bucket"
            row["pct_cam_hour"] = None
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert vehicle counts into road density and congestion labels")
    dp.add_tag_argument(parser)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--centers", type=Path, default=None, help="cnn_centers.csv (optional)")
    parser.add_argument("--cnn-counts", type=Path, default=None, help="cnn_counts.csv (optional)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label-bucket", default="camera_hour", choices=["camera_hour", "global"])
    parser.add_argument("--masks", type=Path, default=None, help="Override the shared road-mask directory")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Run even though the masks have not been human-confirmed (development only)",
    )
    parser.add_argument("--build-plates", action="store_true", help="(Re)build per-camera temporal-median clean plates")
    parser.add_argument("--residual", action="store_true", help="Compute residual_occupancy (needs clean plates; reads every image)")
    parser.add_argument("--residual-tau", type=int, default=25, help="Grayscale difference threshold for the residual signal")
    parser.add_argument(
        "--conflict-residual",
        type=float,
        default=0.15,
        help="Residual above this with almost no detections flags a detector conflict (see docs 4.5)",
    )
    args = parser.parse_args()

    tag = dp.resolve_tag(args.dataset_tag)
    manifest_path = args.manifest or dp.stage_dir(tag, "02_dataset") / "image_manifest.csv"
    detections_path = args.detections or dp.stage_dir(tag, "03_yolo_detection") / "frame_detections.csv"
    centers_path = args.centers or dp.stage_dir(tag, "03b_cnn_count") / "cnn_centers.csv"
    counts_path = args.cnn_counts or dp.stage_dir(tag, "03b_cnn_count") / "cnn_counts.csv"
    output_dir = args.output or dp.stage_dir(tag, "04b_congestion")
    print(f"Dataset tag: {tag}")

    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not detections_path.is_file():
        raise SystemExit(f"Detections not found: {detections_path}")

    manifest = read_manifest(manifest_path)
    cameras = sorted({row.get("camera_id", "") for row in manifest})

    # 缺掩码直接报错并给出该跑的命令，绝不静默 NaN
    missing = [c for c in cameras if load_mask(c) is None]
    if missing:
        raise SystemExit(
            f"缺少道路掩码：{', '.join(missing)}\n"
            f"先运行：python src/04a_build_road_masks.py --review"
        )
    if not args.allow_unreviewed:
        unreviewed = [c for c in cameras if not (rg.load_geometry(c) or rg.RoadGeometry(camera_id=c)).reviewed]
        if unreviewed:
            raise SystemExit(
                f"掩码尚未人工确认：{', '.join(unreviewed)}\n"
                f"看一眼 road_masks/review/<camera>.png（相对项目根），\n"
                f"确认无误后在 <camera>.json 里把 reviewed 改成 true。\n"
                f"（开发期可加 --allow-unreviewed 跳过这个门控）"
            )

    masks = {c: load_mask(c) for c in cameras}
    geometries = {c: rg.load_geometry(c) for c in cameras}

    # 底图与残差信号：把清单按相机分好，便于抽样建底图
    paths_by_camera: dict[str, list[Path]] = defaultdict(list)
    for entry in manifest:
        camera_id = entry.get("camera_id", "")
        resolved = dp.resolve_stored_path(tag, camera_id, entry.get("image_path", ""))
        if resolved.is_file():
            paths_by_camera[camera_id].append(resolved)
    for paths in paths_by_camera.values():
        paths.sort()

    plates: dict[str, np.ndarray | None] = {}
    if args.build_plates or args.residual:
        import cv2

        for camera_id in cameras:
            plate_path = dp.clean_plate_path(camera_id)
            if args.build_plates or not plate_path.is_file():
                plate = build_clean_plate(camera_id, paths_by_camera.get(camera_id, []))
                if plate is not None:
                    plate_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(plate_path), plate)
                    print(f"  clean plate: {camera_id} (median of {min(len(paths_by_camera.get(camera_id, [])), 150)} frames)")
            plates[camera_id] = cv2.imread(str(plate_path), cv2.IMREAD_GRAYSCALE) if plate_path.is_file() else None
        missing_plates = [c for c in cameras if plates.get(c) is None]
        if missing_plates:
            print(f"  警告：这些相机没有底图，残差列将为空：{', '.join(missing_plates)}")

    boxes_by_image = rg.load_box_map(detections_path)
    centers_by_image = read_cnn_centers(centers_path)
    counts = read_cnn_counts(counts_path)
    if not centers_by_image:
        print(f"提示：没有 CNN 中心点文件（{centers_path.name}），cnn 列将为 0")

    rows = []
    for entry in manifest:
        camera_id = entry.get("camera_id", "")
        stored = entry.get("image_path", "")
        mask = masks.get(camera_id)
        if mask is None:
            continue
        resolved = dp.resolve_stored_path(tag, camera_id, stored)
        key = str(resolved)
        boxes = boxes_by_image.get(stored) or boxes_by_image.get(key) or []
        centers = centers_by_image.get(stored) or centers_by_image.get(key) or []

        covered, occ = rg.occupancy_px(mask, boxes)
        geometry = geometries.get(camera_id)
        occ_gnd = None
        if geometry is not None:
            occ_gnd = rg.occupancy_ground(mask, boxes, geometry)[1]

        heights = [y2 - y1 for _c, _conf, _x1, y1, _x2, y2 in boxes]
        confidences = [conf for _c, conf, _x1, _y1, _x2, _y2 in boxes]
        med_h = float(statistics.median(heights)) if heights else 0.0
        med_conf = float(statistics.median(confidences)) if confidences else 0.0
        n_cnn_in = points_in_mask(mask, centers)
        cnn_total = counts.get((camera_id, stored))

        captured = parse_capture_time(stored)
        # 可测性门控：测不到就输出空标签，绝不能当成「畅通」（夜间置信度会跌到 0.10–0.29）
        reasons = []
        if boxes:
            if med_conf < MIN_MEDIAN_CONFIDENCE:
                reasons.append(f"median_conf={med_conf:.3f}<{MIN_MEDIAN_CONFIDENCE}")
            if med_h < MIN_MEDIAN_BOX_HEIGHT:
                reasons.append(f"median_box_h={med_h:.1f}<{MIN_MEDIAN_BOX_HEIGHT}")
        elif not centers:
            reasons.append("no_boxes_and_no_cnn_density")

        # 残差信号：不依赖检测器，所以夜间也能用
        residual = float("nan")
        if args.residual and plates.get(camera_id) is not None:
            import cv2

            image = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                small = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
                mask_small = cv2.resize(mask.astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST) > 0
                residual = residual_occupancy(small, plates[camera_id], mask_small, args.residual_tau)

        # 检测器冲突：画面显示路面与底图明显不同却几乎没检出车 —— 单看已检出框的质量抓不到这种漏检
        if not np.isnan(residual) and residual > args.conflict_residual and len(boxes) <= 2:
            reasons.append(f"detector_conflict(residual={residual:.3f},n_yolo={len(boxes)})")

        measurable = not reasons

        rows.append({
            "camera_id": camera_id,
            "image_path": key,
            "captured_at": captured.isoformat() if captured else "",
            "hour_sgt": sgt_hour(captured),
            "road_px": int(mask.sum()),
            "road_area_frac": round(float(mask.mean()), 5),
            "n_yolo": len(boxes),
            "n_yolo_in_road": sum(
                1 for _c, _conf, x1, y1, x2, y2 in boxes
                if mask[int(np.clip((y1 + y2) / 2, 0, mask.shape[0] - 1)),
                        int(np.clip((x1 + x2) / 2, 0, mask.shape[1] - 1))]
            ),
            "n_cnn_in_road": n_cnn_in,
            "cnn_count": "" if cnn_total is None else cnn_total,
            "covered_px": covered,
            "occ_px": round(occ, 6),
            "occ_gnd": "" if occ_gnd is None else round(occ_gnd, 8),
            "median_box_h_px": round(med_h, 2),
            "median_confidence": round(med_conf, 4),
            "residual_occupancy": "" if np.isnan(residual) else round(residual, 5),
            "measurable": measurable,
            "meas_reasons": "; ".join(reasons),
            "yolo_count": len(boxes),
        })

    if not rows:
        raise SystemExit("No rows produced.")

    thresholds = assign_labels(rows, args.label_bucket)

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "congestion_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 逐相机汇总
    summary = []
    for camera_id in sorted({r["camera_id"] for r in rows}):
        subset = [r for r in rows if r["camera_id"] == camera_id]
        measurable_rows = [r for r in subset if r["measurable"]]
        occs = [r["occ_px"] for r in measurable_rows]
        labels = [r["label"] for r in measurable_rows if r["label"]]
        summary.append({
            "camera_id": camera_id,
            "n_images": len(subset),
            "measurable_frac": round(len(measurable_rows) / len(subset), 4),
            "road_area_frac": subset[0]["road_area_frac"],
            "mean_occ_px": round(float(np.mean(occs)), 5) if occs else "",
            "mean_yolo_count": round(float(np.mean([r["yolo_count"] for r in subset])), 2),
            "congested_frac": round(labels.count(LABEL_CONGESTED) / len(labels), 4) if labels else "",
        })
    with (output_dir / "camera_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    (output_dir / "label_thresholds.json").write_text(
        json.dumps({"bucket": args.label_bucket, "min_bucket_n": MIN_BUCKET_N, "thresholds": thresholds},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps({
            "dataset_tag": tag,
            "n_rows": len(rows),
            "cameras": sorted({r["camera_id"] for r in rows}),
            "label_bucket": args.label_bucket,
            "min_median_box_height": MIN_MEDIAN_BOX_HEIGHT,
            "allow_unreviewed": args.allow_unreviewed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'camera':8s} {'road_frac':>9s} {'measurable':>10s} {'mean_occ':>9s} {'mean_count':>10s}")
    for row in summary:
        print(f"{row['camera_id']:8s} {row['road_area_frac']:>9.3f} {row['measurable_frac']:>10.3f} "
              f"{str(row['mean_occ_px']):>9s} {row['mean_yolo_count']:>10.2f}")
    print(f"\nSaved: {output_dir / 'congestion_metrics.csv'}")


if __name__ == "__main__":
    main()
