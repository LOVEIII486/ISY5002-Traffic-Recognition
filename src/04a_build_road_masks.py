"""从累积的 YOLO 检测框自动生成每个相机的道路掩码（输出到共享的 road_masks/）。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenCV is required. Install it with: pip install opencv-python") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as dp
import road_geometry as rg


IMAGE_SHAPE = (1080, 1920)
# 烧录的 LTA logo 固定在这个角落，里面不可能有路
LOGO_BOX = (0, 0, 200, 100)


def collect_camera_box_map(tags: list[str]) -> dict[str, dict]:
    """把各标签的检测框按相机和图片汇总，并挑一张人工确认用的样图。"""
    per_camera: dict[str, dict] = {}
    for tag in tags:
        detections = dp.stage_dir(tag, "03_yolo_detection") / "frame_detections.csv"
        if not detections.is_file():
            print(f"  [{tag}] 没有检测结果，跳过：{detections}")
            continue
        by_image = rg.load_box_map(detections)
        for image_path, boxes in by_image.items():
            # 目录名就是 camera_id，比解析文件名可靠
            camera_id = Path(image_path).parent.name
            entry = per_camera.setdefault(camera_id, {"by_image": {}, "sample": None, "sample_key": None})
            entry["by_image"][image_path] = boxes
            resolved = dp.resolve_stored_path(tag, camera_id, image_path)
            if resolved.is_file():
                # 后处理的标签胜出（registry 顺序里 2026-week1 更靠后）
                entry["sample"] = resolved
                entry["sample_key"] = image_path
        print(f"  [{tag}] {len(by_image)} 张图")
    return per_camera


def build_coverage_mask(
    per_image_boxes: dict[str, list],
    shape: tuple[int, int] = IMAGE_SHAPE,
    min_coverage_frac: float = 0.02,
    min_component_area: int = 3000,
    dilate_px: int = 8,
    close_px: int = 21,
    open_px: int = 9,
    hull: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """按「逐像素被车辆覆盖的图像比例」阈值化出道路区域，保留所有面积达标的连通域。"""
    n_images = len(per_image_boxes)
    if n_images == 0:
        return np.zeros(shape, dtype=bool), None, {"n_images": 0}

    accumulator = np.zeros(shape, dtype=np.uint16)
    for boxes in per_image_boxes.values():
        accumulator += rg.boxes_to_mask(shape, rg._strip(boxes)).astype(np.uint16)

    coverage = accumulator.astype(np.float32) / n_images
    seed = (coverage >= min_coverage_frac).astype(np.uint8) * 255

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
        seed = cv2.dilate(seed, k)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1,) * 2)
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px * 2 + 1,) * 2)
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, k)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    if count <= 1:
        return np.zeros(shape, dtype=bool), coverage, {"n_images": n_images, "components_kept": 0}

    # 保留所有面积达标的连通域（不是只留最大的那个）
    kept, kept_areas = [], []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            kept.append(label)
            kept_areas.append(area)
    if not kept:
        biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        kept, kept_areas = [biggest], [int(stats[biggest, cv2.CC_STAT_AREA])]

    mask = np.isin(labels, kept)
    if hull:
        canvas = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask = cv2.fillPoly(np.zeros_like(canvas), [cv2.convexHull(np.vstack(contours))], 255) > 0

    mask[LOGO_BOX[1]:LOGO_BOX[3], LOGO_BOX[0]:LOGO_BOX[2]] = False
    diagnostics = {
        "n_images": n_images,
        "components_total": count - 1,
        "components_kept": len(kept),
        "kept_areas": sorted(kept_areas, reverse=True)[:6],
        "coverage_p99": float(np.percentile(coverage, 99)),
    }
    return mask, coverage, diagnostics


def mask_qa(mask: np.ndarray, boxes: list) -> dict:
    """自动 QA：统计检测框中心落在掩码内的比例。"""
    if not boxes:
        return {"center_in_mask_frac": None, "logo_clean": True, "dominant_frac": None}
    h, w = mask.shape[:2]
    inside = 0
    for _c, _conf, x1, y1, x2, y2 in boxes:
        cx = int(np.clip((x1 + x2) / 2.0, 0, w - 1))
        cy = int(np.clip((y1 + y2) / 2.0, 0, h - 1))
        if mask[cy, cx]:
            inside += 1
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8) * 255, connectivity=8)
    area = float(mask.sum())
    dominant = float(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0.0
    return {
        "center_in_mask_frac": inside / len(boxes),
        "logo_clean": not bool(mask[LOGO_BOX[1]:LOGO_BOX[3], LOGO_BOX[0]:LOGO_BOX[2]].any()),
        "dominant_frac": (dominant / area) if area else None,
    }


def write_review_overlay(
    camera_id: str,
    mask: np.ndarray,
    sample_boxes: list,
    sample_image: Path | None,
    coverage: np.ndarray | None = None,
    diagnostics: dict | None = None,
) -> Path | None:
    """输出人工确认叠加图：该帧自己的检测框 + 掩码轮廓 + 覆盖率热力图。"""
    if sample_image is None or not sample_image.is_file():
        return None
    image = cv2.imread(str(sample_image), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    overlay = image.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([0, 200, 255])).astype(np.uint8)
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 3)
    for _c, _conf, x1, y1, x2, y2 in sample_boxes:
        cv2.rectangle(blended, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    header = f"camera {camera_id}  road_frac={mask.mean():.3f}  boxes_in_this_frame={len(sample_boxes)}"
    cv2.rectangle(blended, (0, 0), (blended.shape[1], 90), (0, 0, 0), -1)
    cv2.putText(blended, header, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

    # 右下角插入覆盖率热力图：掩码是从它阈值出来的，人工确认时最该看的就是它
    if coverage is not None:
        heat = np.clip(coverage / max(float(np.percentile(coverage, 99.5)), 1e-6), 0, 1)
        heat = (heat * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        small = cv2.resize(heat, (640, 360), interpolation=cv2.INTER_AREA)
        contours_small = cv2.findContours(
            cv2.resize(mask.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        cv2.drawContours(small, contours_small, -1, (0, 0, 255), 2)
        cv2.putText(small, "detection coverage (mask seed)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        blended[blended.shape[0] - 370:blended.shape[0] - 10, blended.shape[1] - 650:blended.shape[1] - 10] = small

    review_dir = dp.road_mask_dir() / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    out = review_dir / f"{camera_id}_review.png"
    cv2.imwrite(str(out), blended)
    return out


def write_manifest(cameras: list[str], tags: list[str]) -> Path:
    """写相机并集清单，记录各相机出现在哪些标签、掩码是否已确认。"""
    present = {}
    for camera_id in cameras:
        in_tags = [
            tag
            for tag in tags
            if (dp.raw_dir(tag) / camera_id).is_dir()
        ]
        geom = rg.load_geometry(camera_id)
        present[camera_id] = {
            "tags": in_tags,
            "mask": dp.road_mask_path(camera_id).is_file(),
            "reviewed": bool(geom.reviewed) if geom else False,
            "road_area_frac": round(geom.area_frac, 4) if geom else None,
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tags": tags,
        "camera_count": len(cameras),
        "reviewed_count": sum(1 for v in present.values() if v["reviewed"]),
        "cameras": present,
    }
    path = dp.road_mask_dir() / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-camera road masks from accumulated detections")
    parser.add_argument("--dataset-tag", action="append", default=None, help="Repeatable; defaults to every registered tag")
    parser.add_argument("--camera-id", default="all", help="Camera ID, or all")
    parser.add_argument(
        "--min-coverage-frac",
        type=float,
        default=0.005,
        help=(
            "Keep pixels covered by at least this fraction of the camera's images. "
            "Lower = more inclusive. 阈值扫描实测：0.002/0.005/0.02/0.05 对应的 "
            "4713 in-mask = 0.76/0.48/0.37/0.22、4703 = 0.96/0.93/0.86/0.71。"
            "默认偏向「宁多勿漏」——删掉多余的块比补画缺失的路面容易，"
            "而且没有任何单一全局阈值能同时适配 10 个结构迥异的相机，最终靠人工确认兜底。"
        ),
    )
    parser.add_argument("--min-component-area", type=int, default=3000, help="Drop connected components smaller than this many px")
    parser.add_argument("--dilate-px", type=int, default=8)
    parser.add_argument("--close-px", type=int, default=21)
    parser.add_argument("--open-px", type=int, default=9)
    parser.add_argument("--hull", action="store_true", help="Fill the convex hull (over-fills multi-carriageway scenes)")
    parser.add_argument("--review", action="store_true", help="Also write a human-verification overlay per camera")
    parser.add_argument("--check-only", action="store_true", help="Only report the tag/mask coverage matrix")
    args = parser.parse_args()

    tags = args.dataset_tag or dp.existing_tags()
    if not tags:
        raise SystemExit("No dataset tags found. Pass --dataset-tag.")

    if args.check_only:
        print(f"Tags: {tags}")
        cameras = sorted(
            {
                path.name
                for tag in tags
                for path in dp.raw_dir(tag).iterdir()
                if path.is_dir()
            }
        )
        print(f"{'camera':8s} {'mask':6s} {'reviewed':9s} {'area_frac':>9s}  tags")
        for camera_id in cameras:
            geom = rg.load_geometry(camera_id)
            in_tags = [tag for tag in tags if (dp.raw_dir(tag) / camera_id).is_dir()]
            print(
                f"{camera_id:8s} {str(dp.road_mask_path(camera_id).is_file()):6s} "
                f"{str(bool(geom.reviewed) if geom else False):9s} "
                f"{(f'{geom.area_frac:.3f}' if geom else '-'):>9s}  {','.join(in_tags)}"
            )
        write_manifest(cameras, tags)
        print(f"\nUnion: {len(cameras)} cameras")
        return

    print(f"Collecting detections from tags: {tags}")
    per_camera = collect_camera_box_map(tags)
    if not per_camera:
        raise SystemExit("No detections found. Run 03_run_yolo_detection.py first.")

    wanted = sorted(per_camera) if args.camera_id == "all" else [args.camera_id]
    dp.road_mask_dir().mkdir(parents=True, exist_ok=True)

    for camera_id in wanted:
        entry = per_camera.get(camera_id)
        if not entry or not entry["by_image"]:
            print(f"[{camera_id}] 无检测，跳过")
            continue
        by_image = entry["by_image"]
        all_boxes = [box for boxes in by_image.values() for box in boxes]
        mask, coverage, diag = build_coverage_mask(
            by_image,
            IMAGE_SHAPE,
            min_coverage_frac=args.min_coverage_frac,
            min_component_area=args.min_component_area,
            dilate_px=args.dilate_px,
            close_px=args.close_px,
            open_px=args.open_px,
            hull=args.hull,
        )
        if not mask.any():
            print(f"[{camera_id}] 掩码为空，跳过")
            continue

        cv2.imwrite(str(dp.road_mask_path(camera_id)), mask.astype(np.uint8) * 255)

        geom = rg.RoadGeometry(
            camera_id=camera_id,
            b=1.0,
            c=0.0,  # 灭线默认置顶；不报拟合值，人工确认时再定
            w_min=1.0,
            area_px=int(mask.sum()),
            area_frac=float(mask.mean()),
        )
        slope, r2, n = rg.fit_size_law(all_boxes, geom)
        # 尺寸律与灭线选择无关（y_h=0 时 u 就是行号），所以 w_min 仍可用
        geom = replace(
            geom,
            w_min=rg.w_min_from_slope(slope),
            size_slope=slope,
            size_r2=r2,
            method="coverage" + ("+hull" if args.hull else ""),
        )
        geom = replace(geom, area_eff=float((mask * rg.weight_map(mask.shape, geom, exponent=3.0)).sum()))
        rg.save_geometry(geom)

        qa = mask_qa(mask, all_boxes)
        in_mask = "n/a" if qa["center_in_mask_frac"] is None else f"{qa['center_in_mask_frac']:.3f}"
        print(
            f"[{camera_id}] area_frac={geom.area_frac:.3f} boxes_center_in={in_mask} "
            f"components={diag['components_kept']}/{diag['components_total']} "
            f"areas={diag['kept_areas'][:3]} "
            f"logo_clean={qa['logo_clean']} size R2={r2:.3f}"
        )

        if args.review:
            sample_key = entry.get("sample_key")
            sample_boxes = by_image.get(sample_key, []) if sample_key else []
            overlay = write_review_overlay(camera_id, mask, sample_boxes, entry["sample"], coverage, diag)
            if overlay:
                print(f"          review overlay -> {overlay}")
            else:
                print(f"          [warn] 无可用样图，未生成叠加图")

    manifest = write_manifest(sorted(per_camera), tags)
    print(f"\nMasks in: {dp.road_mask_dir()}")
    print(f"Manifest: {manifest}")
    print("提示：掩码 currently reviewed=false。看一眼 review/*.png 确认无误后，")
    print("     在对应 <camera_id>.json 里把 reviewed 改成 true，04b 才肯跑。")


if __name__ == "__main__":
    main()
