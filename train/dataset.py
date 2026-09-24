"""TRANCOS dataset loading and density-map generation from point annotations.

Each TRANCOS annotation is a text file of "x y" pixel coordinates, one vehicle
point per line. We turn those points into a density map by blurring a unit
impulse per point with a Gaussian (sum-preserving), then downsample it to the
network's output resolution by summing blocks. The sum of the density map is
therefore approximately the number of vehicles.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def points_to_density(points, height: int, width: int, sigma: float) -> np.ndarray:
    """Blur one unit impulse per point into a density map whose sum == n_points."""
    density = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            density[yi, xi] += 1.0
    if sigma > 0:
        ksize = int(round(sigma * 6)) | 1  # odd, captures ~3*sigma tail
        density = cv2.GaussianBlur(density, (ksize, ksize), sigmaX=sigma)
    return density


def downsample_density(density: np.ndarray, factor: int) -> np.ndarray:
    """Sum `factor` x `factor` blocks; preserves the total count."""
    h = (density.shape[0] // factor) * factor
    w = (density.shape[1] // factor) * factor
    d = density[:h, :w]
    return d.reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))


class TrancosDataset(Dataset):
    """Loads TRANCOS images + point annotations and produces (image, density, count)."""

    def __init__(
        self,
        data_dir,
        split: str = "training",
        input_size=(480, 640),
        sigma: float = 15.0,
        factor: int = 8,
        augment: bool = False,
        seed=None,
        use_roi_mask: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.h, self.w = input_size
        self.sigma = sigma
        self.factor = factor
        self.augment = augment
        self.use_roi_mask = use_roi_mask
        self.rng = np.random.default_rng(seed)

        assert self.h % factor == 0 and self.w % factor == 0, (
            f"input_size {input_size} must be divisible by factor {factor}"
        )

        split_file = self.data_dir / "image_sets" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        self.names = self._read_split(split_file)

    def _read_split(self, split_file: Path) -> list[str]:
        """Keep only images that have both a jpg and a txt annotation."""
        valid = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name:
                continue
            img = self.data_dir / "images" / name
            txt = self.data_dir / "images" / (Path(name).stem + ".txt")
            if img.is_file() and txt.is_file():
                valid.append(name)
        return valid

    def __len__(self) -> int:
        return len(self.names)

    def _load_points(self, name: str) -> list[tuple[float, float]]:
        txt = self.data_dir / "images" / (Path(name).stem + ".txt")
        points = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        return points

    def _load_mask(self, name: str) -> np.ndarray:
        """Load the official ROI mask (480x640 binary) for evaluation."""
        mask_path = self.data_dir / "images" / (Path(name).stem + "mask.mat")
        return sio.loadmat(str(mask_path))["BW"].astype(np.float32)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = cv2.imread(str(self.data_dir / "images" / name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        points = self._load_points(name)
        mask = None
        if self.use_roi_mask:
            roi = self._load_mask(name)
            points = [
                (x, y) for x, y in points
                if 0 <= int(y) < orig_h and 0 <= int(x) < orig_w and roi[int(y), int(x)] > 0
            ]
            mask = roi

        # Scale point coordinates to the target input size.
        sx, sy = self.w / orig_w, self.h / orig_h
        points = [(x * sx, y * sy) for x, y in points]

        if self.augment and self.rng.random() < 0.5:
            img = img[:, ::-1, :]
            points = [(self.w - 1 - x, y) for x, y in points]
            if mask is not None:
                mask = mask[:, ::-1]

        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        sigma_eff = self.sigma * (self.w / 640.0)
        density = points_to_density(points, self.h, self.w, sigma_eff)
        density = downsample_density(density, self.factor)

        if mask is not None:
            mask_small = cv2.resize(mask, (self.w // self.factor, self.h // self.factor), interpolation=cv2.INTER_NEAREST)
            mask_t = torch.from_numpy(np.ascontiguousarray(mask_small)).unsqueeze(0).float()
        else:
            mask_t = torch.ones(1, self.h // self.factor, self.w // self.factor)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        density_t = torch.from_numpy(np.ascontiguousarray(density)).unsqueeze(0).float()
        count = float(len(points))
        return img_t, density_t, torch.tensor([count], dtype=torch.float32), mask_t
