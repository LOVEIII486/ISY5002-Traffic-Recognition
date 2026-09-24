"""Run YOLO vehicle detection for camera-level traffic analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from ultralytics import YOLO

import dataset_paths as dp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 权重与数据集无关，保持固定；输入/输出目录改由 --dataset-tag 推导。
DEFAULT_MODEL = PROJECT_ROOT / "yolo" / "yolo26x.pt"
DEFAULT_CLASS_NAMES = {"car", "motorcycle", "bus", "truck", "bicycle"}


def download_model(model_path: Path) -> None:
    """让 Ultralytics 自动下载官方模型到 yolo 目录。"""
    if model_path.is_file():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.parent != PROJECT_ROOT / "yolo":
        raise FileNotFoundError(f"Model does not exist and automatic download only supports the yolo directory: {model_path}")
    model_url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_path.name}"
    temporary_path = model_path.with_suffix(".pt.download")
    print(f"Model not found locally. Downloading: {model_path.name}")
    try:
        urllib.request.urlretrieve(model_url, temporary_path)
        temporary_path.replace(model_path)
    except Exception as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(f"Model download failed: {model_url}") from exc


def load_rows(manifest_path: Path, camera_id: str, limit: int) -> list[dict[str, str]]:
    """读取图片清单并选择摄像头图片。"""
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if camera_id == "all" or row.get("camera_id") == camera_id]
    rows.sort(key=lambda row: (row.get("camera_id", ""), row["image_path"]))
    selected = rows[:limit] if limit > 0 else rows
    if not selected:
        raise RuntimeError(f"No images found for camera {camera_id}")
    for row in selected:
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
    return selected


def make_image_name(image_path: Path) -> str:
    """生成不会冲突的标记图文件名。"""
    token = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:8]
    return f"{image_path.stem}_{token}{image_path.suffix.lower()}"


def get_vehicle_class_ids(model: YOLO) -> tuple[list[int], dict[int, str]]:
    """从模型自身类别映射中查找车辆类别。"""
    names = model.names
    name_map = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {i: str(v) for i, v in enumerate(names)}
    class_ids = sorted(k for k, v in name_map.items() if v.lower() in DEFAULT_CLASS_NAMES)
    if not class_ids:
        raise RuntimeError(f"No configured vehicle classes found in model names: {name_map}")
    return class_ids, name_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO vehicle detection on traffic images")
    dp.add_tag_argument(parser)
    parser.add_argument("--input-manifest", type=Path, default=None, help="Image manifest (default: results/<tag>/02_dataset/image_manifest.csv)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: results/<tag>/03_yolo_detection)")
    parser.add_argument("--camera-id", default="2701", help="Camera ID; use all to process every camera")
    parser.add_argument("--limit", type=int, default=5, help="Maximum images; use 0 for all selected images")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-annotated", dest="save_annotated", action="store_true", default=True, help="Save annotated images (enabled by default)")
    parser.add_argument("--no-save-annotated", dest="save_annotated", action="store_false", help="Disable annotated image output")
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag)
    input_manifest = args.input_manifest if args.input_manifest is not None else dp.stage_dir(tag, "02_dataset") / "image_manifest.csv"
    output = args.output if args.output is not None else dp.stage_dir(tag, "03_yolo_detection")
    print(f"Dataset tag: {tag}")
    if not input_manifest.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {input_manifest}")
    if args.limit < 0:
        raise ValueError("limit must not be negative")

    download_model(args.model)
    model = YOLO(str(args.model))
    class_ids, class_name_map = get_vehicle_class_ids(model)
    rows = load_rows(input_manifest, args.camera_id, args.limit)
    output.mkdir(parents=True, exist_ok=True)
    annotated_dir = output / "annotated"
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    detection_rows: list[dict] = []
    image_results: list[dict] = []
    failed_images: list[dict] = []
    total_time = 0.0
    for row in rows:
        image_path = Path(row["image_path"])
        started = time.perf_counter()
        try:
            result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, classes=class_ids, save=False, verbose=False)[0]
            elapsed_ms = (time.perf_counter() - started) * 1000
            total_time += elapsed_ms
            annotated_image = result.orig_img.copy() if args.save_annotated else None
            image_detections = []
            if result.boxes is not None:
                for class_id, confidence, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
                    detection = {
                        "camera_id": row["camera_id"], "class_id": int(class_id), "class_name": class_name_map[int(class_id)],
                        "confidence": round(float(confidence), 6), "x1": round(float(box[0]), 2), "y1": round(float(box[1]), 2),
                        "x2": round(float(box[2]), 2), "y2": round(float(box[3]), 2), "image_path": str(image_path.resolve()),
                    }
                    detection_rows.append(detection)
                    image_detections.append(detection)
                    if args.save_annotated:
                        color = (0, 220, 0)
                        cv2.rectangle(annotated_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
                        label = f"{detection['class_name']} {detection['confidence']:.2f}"
                        cv2.putText(annotated_image, label, (int(box[0]), max(25, int(box[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            if args.save_annotated:
                cv2.imwrite(str(annotated_dir / make_image_name(image_path)), annotated_image)
            image_results.append({"camera_id": row["camera_id"], "image_path": str(image_path.resolve()), "vehicle_count": len(image_detections), "inference_time_ms": round(elapsed_ms, 2)})
            print(f"Processed {image_path.name}: {len(image_detections)} vehicles, {elapsed_ms:.1f} ms")
        except Exception as exc:
            failed_images.append({"image_path": str(image_path.resolve()), "error": f"{type(exc).__name__}: {exc}"})
            print(f"Failed to process {image_path.name}: {exc}")

    fields = ["camera_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "image_path"]
    with (output / "frame_detections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detection_rows)
    metadata = {"dataset_tag": tag, "model": str(args.model.resolve()), "camera_id": args.camera_id, "requested_images": len(rows), "processed_images": len(image_results), "failed_images": failed_images, "total_detections": len(detection_rows), "average_inference_time_ms": round(total_time / len(image_results), 2) if image_results else None, "imgsz": args.imgsz, "confidence": args.conf, "iou": args.iou, "device": args.device, "class_ids": class_ids, "classes": [class_name_map[i] for i in class_ids], "annotated_images_saved": args.save_annotated}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
