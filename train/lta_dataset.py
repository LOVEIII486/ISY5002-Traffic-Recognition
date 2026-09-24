"""LTA pseudo-label dataset for domain-adapting the counting CNN.

YOLO's bounding-box centers on the project's own LTA images serve as point
supervision. Each image is resized to a 16:9 input and converted into a density
map, mirroring the TRANCOS training setup but on the deployment domain. This
teaches the model to localise density only where LTA vehicles actually are.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import downsample_density, points_to_density


class LtaPseudoDataset(Dataset):
    def __init__(
        self,
        detections_csv,
        image_manifest,
        input_size=(360, 640),
        sigma=15.0,
        factor=8,
        min_conf=0.25,
        train=True,
        val_fraction=0.1,
        augment=False,
        seed=42,
    ) -> None:
        self.h, self.w = input_size
        self.sigma = sigma
        self.factor = factor
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        assert self.h % factor == 0 and self.w % factor == 0, "input_size must be divisible by factor"

        # image_path -> list of (cx, cy) box centers in original image pixels.
        boxes_by_image: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
        with open(detections_csv, newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                try:
                    conf = float(row.get("confidence", 0.0))
                except (TypeError, ValueError):
                    continue
                if conf < min_conf:
                    continue
                try:
                    x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                except (KeyError, TypeError, ValueError):
                    continue
                boxes_by_image[row["image_path"]].append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        samples = []
        with open(image_manifest, newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                image_path = row["image_path"]
                if not Path(image_path).is_file():
                    continue
                samples.append((image_path, boxes_by_image.get(image_path, [])))

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(samples))
        n_val = int(len(samples) * val_fraction)
        val_indices = set(order[:n_val].tolist())
        keep = [i for i in range(len(samples)) if (i in val_indices) != train]
        self.samples = [samples[i] for i in keep]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, points_orig = self.samples[idx]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        sx, sy = self.w / orig_w, self.h / orig_h
        points = [(x * sx, y * sy) for x, y in points_orig]

        if self.augment and self.rng.random() < 0.5:
            img = img[:, ::-1, :]
            points = [(self.w - 1 - x, y) for x, y in points]

        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        sigma_eff = self.sigma * (self.w / 640.0)
        density = points_to_density(points, self.h, self.w, sigma_eff)
        density = downsample_density(density, self.factor)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        density_t = torch.from_numpy(np.ascontiguousarray(density)).unsqueeze(0).float()
        mask_t = torch.ones(1, self.h // self.factor, self.w // self.factor)
        return img_t, density_t, torch.tensor([float(len(points))], dtype=torch.float32), mask_t
