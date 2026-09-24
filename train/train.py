"""Train the point-supervised density-regression counting model on TRANCOS.

Run from anywhere::

    python train.py --data-dir TRANCOS_v3 --epochs 80

The script trains on the official ``training`` split, validates on the official
``validation`` split, and saves ``best.pt`` (lowest val MAE) and ``last.pt``
plus a ``config.json`` into the output directory.
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

from dataset import TrancosDataset
from model import CountNet

ROOT = Path(__file__).resolve().parent

# Density-map pixels are tiny (~count / 4800), so a raw per-pixel MSE produces
# ~1e-4 losses and a near-zero gradient that lets the model collapse to an
# all-zero output. Scaling both maps up by this factor restores a healthy
# gradient. The count is still read from the *unscaled* sum, so the
# "sum == count" contract is untouched.
DENSITY_SCALE = 100.0


def build_loaders(args):
    train_ds = TrancosDataset(
        args.data_dir, "training", input_size=(args.input_h, args.input_w),
        sigma=args.sigma, factor=args.factor, augment=True,
    )
    val_ds = TrancosDataset(
        args.data_dir, "validation", input_size=(args.input_h, args.input_w),
        sigma=args.sigma, factor=args.factor, augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
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
    mape = float(np.mean(np.abs(preds - gts) / (gts + 1e-6))) * 100.0
    return {"mae": mae, "rmse": rmse, "mape": mape}


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
    tmp.replace(path)  # atomic: Ctrl+C mid-save cannot corrupt an existing checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the counting CNN on TRANCOS")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "TRANCOS_v3")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--input-h", type=int, default=480)
    parser.add_argument("--input-w", type=int, default=640)
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--count-weight", type=float, default=1.0,
                        help="weight of the relative count-consistency loss (0 disables)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    if args.device != "cpu" and not use_cuda:
        print("CUDA not available, falling back to CPU")

    train_loader, val_loader = build_loaders(args)
    model = CountNet(base=args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    start_epoch = 0
    best_mae = float("inf")
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
        start_epoch = ckpt.get("epoch", 0)
        best_mae = ckpt.get("val_mae", best_mae)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best val MAE {best_mae:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "in_channels": 3,
        "base": args.base,
        "input_size": [args.input_h, args.input_w],
        "sigma": args.sigma,
        "factor": args.factor,
    }

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} train", leave=False)
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
                count_loss = ((pred_sum - gt).abs() / (gt + 1.0)).mean()
                loss = loss + args.count_weight * count_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)
            n += img.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        train_loss = total_loss / n

        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["mae"])
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train loss {train_loss:.6f} | "
            f"val MAE {metrics['mae']:.2f} RMSE {metrics['rmse']:.2f} "
            f"MAPE {metrics['mape']:.1f}% | lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, epoch + 1, best_mae, config)
        save_checkpoint(args.out_dir / "last.pt", model, optimizer, epoch + 1, best_mae, config)

    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Training done. Best val MAE {best_mae:.2f}. Checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
