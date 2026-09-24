"""Run the self-trained counting CNN alongside YOLO and record per-image counts.

This is the CNN counterpart of stage 03. It reads the same image manifest and
writes one vehicle count per image using the point-supervised density model
trained in ``train/``. Stage 04 merges these counts with the YOLO detections so
both models' results live side by side in ``traffic_metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "train"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TRAIN_DIR))

import dataset_paths as dp  # noqa: E402
from predict import centers_to_image, extract_centers, load_model, predict_density  # noqa: E402

# CNN 权重按项目共享（不随数据集切分）；输入/输出目录由 --dataset-tag 推导。
#
# 默认必须是做过 LTA 领域自适应微调的 best_lta.pt，而不是只在 TRANCOS 上训练的 best.pt。
# best.pt 直接用在 LTA 图上会把路面的烧录文字（路名、摄像头编号）和纹理当成车辆密度：
# 在 2026-week1 的 160 张样本上实测平均多报 4.77 倍（MAE 40.5、r=+0.23），
# 而 best_lta.pt 是 1.11 倍（MAE 5.8、r=+0.72）。
DEFAULT_WEIGHTS = TRAIN_DIR / "weights" / "best_lta.pt"


def load_rows(manifest_path: Path, camera_id: str, limit: int) -> list[dict[str, str]]:
    """读取图片清单并选择摄像头图片。"""
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if camera_id == "all" or row.get("camera_id") == camera_id]
    rows.sort(key=lambda row: (row.get("camera_id", ""), row.get("image_path", "")))
    selected = rows[:limit] if limit > 0 else rows
    if not selected:
        raise RuntimeError(f"No images found for camera {camera_id}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the counting CNN on traffic images")
    dp.add_tag_argument(parser)
    parser.add_argument("--input-manifest", type=Path, default=None, help="Image manifest (default: results/<tag>/02_dataset/image_manifest.csv)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: results/<tag>/03b_cnn_count)")
    parser.add_argument("--camera-id", default="all", help="Camera ID; use all to process every camera")
    parser.add_argument("--limit", type=int, default=0, help="Maximum images; use 0 for all selected images")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag)
    input_manifest = args.input_manifest if args.input_manifest is not None else dp.stage_dir(tag, "02_dataset") / "image_manifest.csv"
    output = args.output if args.output is not None else dp.stage_dir(tag, "03b_cnn_count")
    print(f"Dataset tag: {tag}")

    if not args.weights.is_file():
        raise FileNotFoundError(
            f"CNN weights not found: {args.weights}\n"
            "  best.pt      trained on TRANCOS only        (train/train.py)         — overcounts LTA ~4.8x\n"
            "  best_lta.pt  domain-adapted to LTA imagery  (train/finetune_lta.py)  — what this stage needs"
        )
    if not input_manifest.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {input_manifest}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.weights, device)
    input_size = tuple(config.get("input_size", [480, 640]))
    factor = config.get("factor", 8)

    rows = load_rows(input_manifest, args.camera_id, args.limit)
    output.mkdir(parents=True, exist_ok=True)

    counts: list[dict] = []
    center_rows: list[dict] = []
    failed: list[dict] = []
    total_time = 0.0
    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            failed.append({"image_path": str(image_path), "error": "file_not_found"})
            continue
        started = time.perf_counter()
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("could not read image")
            density = predict_density(model, img, device, input_size)
            count = float(density.clip(min=0).sum())
            centers = centers_to_image(extract_centers(density), input_size, factor, img.shape[:2])
            elapsed_ms = (time.perf_counter() - started) * 1000
            total_time += elapsed_ms
            counts.append({"camera_id": row["camera_id"], "image_path": str(image_path.resolve()), "cnn_count": round(count, 2)})
            for cx, cy in centers:
                center_rows.append({"image_path": str(image_path.resolve()), "x": cx, "y": cy})
            if index % 500 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)} images (last count {count:.1f}, {len(centers)} points)")
        except Exception as exc:
            failed.append({"image_path": str(image_path.resolve()), "error": f"{type(exc).__name__}: {exc}"})
            print(f"Failed to process {image_path.name}: {exc}")

    with (output / "cnn_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["camera_id", "image_path", "cnn_count"])
        writer.writeheader()
        writer.writerows(counts)

    with (output / "cnn_centers.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "x", "y"])
        writer.writeheader()
        writer.writerows(center_rows)

    metadata = {
        "dataset_tag": tag,
        "weights": str(args.weights.resolve()),
        "model_config": config,
        "device": str(device),
        "requested_images": len(rows),
        "processed_images": len(counts),
        "failed_images": failed,
        "average_inference_time_ms": round(total_time / len(counts), 2) if counts else None,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(counts)} counts to {output / 'cnn_counts.csv'}")


if __name__ == "__main__":
    main()
