"""Inference and evaluation for the trained counting model.

Three modes:

1. Single image (prints the predicted count):
       python predict.py --weights weights/best.pt --image path/to.jpg

2. Manifest mode (writes camera_id,image_path,cnn_count CSV for the pipeline):
       python predict.py --weights weights/best.pt --manifest results/02_dataset/image_manifest.csv

3. TRANCOS split evaluation (reports MAE/RMSE/MAPE against ground truth):
       python predict.py --weights weights/best.pt --eval-split test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import TrancosDataset
from model import CountNet

ROOT = Path(__file__).resolve().parent


def build_model(config: dict) -> CountNet:
    return CountNet(in_channels=config.get("in_channels", 3), base=config.get("base", 64))


def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    model = build_model(ckpt.get("config", {}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt.get("config", {})


def _preprocess(image_bgr, input_size):
    h, w = input_size
    img = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


@torch.no_grad()
def predict_density(model, image_bgr, device, input_size) -> np.ndarray:
    """Return the raw density map (H/f x W/f) for an image."""
    x = _preprocess(image_bgr, input_size).to(device)
    return model(x)[0, 0].cpu().numpy()


@torch.no_grad()
def predict_count(model, image_bgr, device, input_size) -> float:
    return float(predict_density(model, image_bgr, device, input_size).clip(min=0).sum())


def extract_centers(density: np.ndarray, threshold=None) -> list[tuple[int, int]]:
    """Find local maxima in a density map; returns [(x, y), ...] in map coords."""
    if density.max() <= 0:
        return []
    if threshold is None:
        threshold = 0.3 * float(density.max())
    padded = np.pad(density, 1, mode="constant", constant_values=0.0)
    peak = (
        (density >= padded[:-2, 1:-1]) & (density >= padded[2:, 1:-1])
        & (density >= padded[1:-1, :-2]) & (density >= padded[1:-1, 2:])
        & (density >= padded[:-2, :-2]) & (density >= padded[:-2, 2:])
        & (density >= padded[2:, :-2]) & (density >= padded[2:, 2:])
    )
    ys, xs = np.where(peak & (density >= threshold))
    return list(zip(xs.tolist(), ys.tolist()))


def centers_to_image(coords, input_size, factor, orig_shape) -> list[tuple[int, int]]:
    """Map density-map coords (x, y) back to original image pixels."""
    h, w = input_size
    orig_h, orig_w = orig_shape
    scale_x = factor * (orig_w / w)
    scale_y = factor * (orig_h / h)
    return [(round((x + 0.5) * scale_x), round((y + 0.5) * scale_y)) for x, y in coords]


@torch.no_grad()
def evaluate(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, gts = [], []
    for img, _, count, mask in loader:
        img = img.to(device)
        mask = mask.to(device)
        pred = (model(img) * mask).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()
        preds.extend(pred.tolist())
        gts.extend(count.squeeze(1).cpu().numpy().tolist())
    preds = np.array(preds)
    gts = np.array(gts)
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    mape = float(np.mean(np.abs(preds - gts) / (gts + 1e-6))) * 100.0
    return {"mae": mae, "rmse": rmse, "mape": mape, "n": int(gts.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference/eval for the counting CNN")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image", type=Path, default=None, help="single image -> print count")
    parser.add_argument("--manifest", type=Path, default=None, help="image_manifest.csv (camera_id,image_path)")
    parser.add_argument("--output", type=Path, default=None, help="output CSV for manifest mode")
    parser.add_argument("--eval-split", default=None, choices=["training", "validation", "test", "trainval"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "TRANCOS_v3")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.weights, device)
    input_size = tuple(config.get("input_size", [480, 640]))
    print(f"Loaded {args.weights}; input {input_size}")

    if args.image is not None:
        img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        print(f"Predicted count: {predict_count(model, img, device, input_size):.2f}")

    if args.manifest is not None:
        output = args.output or (Path(args.manifest).with_name("cnn_counts.csv"))
        rows = []
        with open(args.manifest, newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                img_path = row.get("image_path")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"skip missing image: {img_path}")
                    continue
                count = predict_count(model, img, device, input_size)
                rows.append({
                    "camera_id": row.get("camera_id", ""),
                    "image_path": img_path,
                    "cnn_count": round(count, 2),
                })
        with open(output, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["camera_id", "image_path", "cnn_count"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} counts to {output}")

    if args.eval_split is not None:
        dataset = TrancosDataset(
            args.data_dir, args.eval_split, input_size=input_size,
            sigma=config.get("sigma", 15.0), factor=config.get("factor", 8),
            use_roi_mask=True,
        )
        metrics = evaluate(model, dataset, device, args.batch_size)
        print(
            f"{args.eval_split} (official ROI mask): MAE {metrics['mae']:.2f} "
            f"RMSE {metrics['rmse']:.2f} MAPE {metrics['mape']:.1f}% (n={metrics['n']})"
        )


if __name__ == "__main__":
    main()
