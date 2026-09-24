"""Fine-tune the TRANCOS-trained counting CNN on LTA via YOLO pseudo-points.

Domain adaptation: loads ``best.pt`` and continues training on the project's own
LTA images using YOLO box centers as point labels. The adapted weights are saved
to SEPARATE files (``best_lta.pt`` / ``last_lta.pt``) so the TRANCOS checkpoint
is left untouched. The reported val MAE is |CNN count - YOLO count| on a held-out
10% of LTA images — exactly the gap the demo cares about.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataset_paths as dp  # noqa: E402
from lta_dataset import LtaPseudoDataset
from model import CountNet

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DENSITY_SCALE = 100.0

DEFAULT_PRETRAINED = ROOT / "weights" / "best.pt"


def build_loaders(args):
    common = dict(
        detections_csv=args.detections,
        image_manifest=args.manifest,
        input_size=(args.input_h, args.input_w),
        sigma=args.sigma,
        factor=args.factor,
        min_conf=args.min_conf,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_ds = LtaPseudoDataset(**common, train=True, augment=True)
    val_ds = LtaPseudoDataset(**common, train=False, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for img, _, count, _mask in loader:
        img = img.to(device)
        pred = model(img).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()
        preds.extend(pred.tolist())
        gts.extend(count.squeeze(1).cpu().numpy().tolist())
    preds = np.array(preds)
    gts = np.array(gts)
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    return {"mae": mae, "rmse": rmse}


def save_checkpoint(path, model, optimizer, epoch, val_mae, config):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_mae": val_mae,
        "config": config,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the counting CNN on LTA (YOLO pseudo-labels)")
    dp.add_tag_argument(parser)
    parser.add_argument("--detections", type=Path, default=None, help="YOLO detection CSV (default: results/<tag>/03_yolo_detection/frame_detections.csv)")
    parser.add_argument("--manifest", type=Path, default=None, help="Image manifest CSV (default: results/<tag>/02_dataset/image_manifest.csv)")
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--out-dir", type=Path, default=None, help="Weights output directory (default: train/weights/<tag>)")
    parser.add_argument("--input-h", type=int, default=360, help="16:9 input height")
    parser.add_argument("--input-w", type=int, default=640, help="16:9 input width")
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--min-conf", type=float, default=0.25, help="ignore YOLO boxes below this confidence")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5, help="fine-tuning learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--count-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag)
    # 输入按标签取；权重默认写到 train/weights/<tag>/，绝不动共享的 best_lta.pt
    # （它被 train/.gitignore 忽略、无法从 git 恢复，而 09 的 demo 依赖它）。
    args.detections = args.detections if args.detections is not None else dp.stage_dir(tag, "03_yolo_detection") / "frame_detections.csv"
    args.manifest = args.manifest if args.manifest is not None else dp.stage_dir(tag, "02_dataset") / "image_manifest.csv"
    args.out_dir = args.out_dir if args.out_dir is not None else ROOT / "weights" / tag
    print(f"Dataset tag: {tag}")
    print(f"Weights output directory: {args.out_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not args.pretrained.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {args.pretrained}")
    if not args.detections.is_file():
        raise FileNotFoundError(f"Detections CSV not found: {args.detections}")

    train_loader, val_loader = build_loaders(args)

    ckpt = torch.load(args.pretrained, map_location=device)
    pretrain_config = ckpt.get("config", {})
    model = CountNet(base=pretrain_config.get("base", 64)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded pretrained {args.pretrained} (source val MAE {ckpt.get('val_mae', 'N/A')})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    start_epoch = 0
    best_mae = float("inf")
    if args.resume and Path(args.resume).is_file():
        resume = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume.get("optimizer_state_dict", optimizer.state_dict()))
        start_epoch = resume.get("epoch", 0)
        best_mae = resume.get("val_mae", best_mae)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "in_channels": 3,
        "base": pretrain_config.get("base", 64),
        "input_size": [args.input_h, args.input_w],
        "sigma": args.sigma,
        "factor": args.factor,
    }

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} finetune", leave=False)
        for img, density, count, _mask in pbar:
            img = img.to(device)
            density = density.to(device)
            count = count.to(device)
            optimizer.zero_grad()
            pred = model(img)
            density_loss = F.mse_loss(pred * DENSITY_SCALE, density * DENSITY_SCALE)
            loss = density_loss
            if args.count_weight > 0:
                pred_sum = pred.sum(dim=(1, 2, 3))
                gt = count.squeeze(1)
                # Normalised absolute count error with a +1 floor so gt=0 images
                # (YOLO found no vehicles) cannot explode the gradient.
                count_loss = ((pred_sum - gt).abs() / (gt + 1.0)).mean()
                loss = loss + args.count_weight * count_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)
            n += img.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        train_loss = total_loss / n

        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["mae"])
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train loss {train_loss:.4f} | "
            f"val MAE |CNN-YOLO| {metrics['mae']:.2f} RMSE {metrics['rmse']:.2f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            save_checkpoint(args.out_dir / "best_lta.pt", model, optimizer, epoch + 1, best_mae, config)
        save_checkpoint(args.out_dir / "last_lta.pt", model, optimizer, epoch + 1, best_mae, config)

    (args.out_dir / "config_lta.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Fine-tune done. Best val MAE {best_mae:.2f}. Saved best_lta.pt / last_lta.pt in {args.out_dir}")


if __name__ == "__main__":
    main()
