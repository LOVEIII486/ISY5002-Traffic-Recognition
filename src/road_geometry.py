"""道路几何：掩码、灭线、透视加权与车辆占用率（推导与取舍见 docs/PROJECT_ARCHITECTURE.md 4.5）。"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenCV is required. Install it with: pip install opencv-python") from exc

import dataset_paths as dp


# 车辆长度/宽度比的经验值。只用于把「覆盖面积」换算成「等效车道占用面积」，
# 因为检测框给的是外接矩形而不是车辆本身的形状。
VEHICLE_ASPECT = 3.0


@dataclass(frozen=True)
class RoadGeometry:
    """一个相机的道路几何。灭线用 ``a*x + b*y + c = 0`` 表示，``(a, b)`` 已单位化。"""

    camera_id: str
    a: float = 0.0
    b: float = 1.0
    c: float = 0.0          # 水平灭线时 c = -y_horizon
    w_min: float = 1.0      # 检测器尺寸下限，单位同为「垂直像素」
    area_px: int = 0
    area_frac: float = 0.0
    area_eff: float = 0.0   # sum(mask / max(w, w_min)**3)
    n_lanes: int | None = None
    reviewed: bool = False
    method: str = ""
    size_slope: float = 0.0
    size_r2: float = 0.0

    @property
    def horizon_row(self) -> float | None:
        """水平灭线时的灭线行号；有倾斜时无单一取值。"""
        if abs(self.b) < 1e-9:
            return None
        return -self.c / self.b


def horizon_distance(shape, geom: RoadGeometry) -> np.ndarray:
    """每个像素到灭线的有符号垂直距离 u（灭线下方为正）。"""
    h, w = shape[:2]
    norm = math.hypot(geom.a, geom.b) or 1.0
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    return (geom.a * xs + geom.b * ys + geom.c) / norm


def weight_map(shape, geom: RoadGeometry, exponent: float = 3.0, w_min: float | None = None) -> np.ndarray:
    """(1 / max(u, w_min)) ** exponent，float32 HxW。"""
    u = horizon_distance(shape, geom)
    floor = geom.w_min if w_min is None else w_min
    return (1.0 / np.maximum(u, max(floor, 1e-6))) ** exponent


def boxes_to_mask(shape, boxes) -> np.ndarray:
    """把一组 (x1, y1, x2, y2) 四元组栅格化成 bool 掩码（重叠只算一次）。"""
    canvas = np.zeros(shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            canvas,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            1,
            thickness=-1,
        )
    return canvas.astype(bool)


def _strip(boxes) -> list[tuple[float, float, float, float]]:
    """六元组 ``(class_name, confidence, x1, y1, x2, y2)`` → 四元组。"""
    return [(x1, y1, x2, y2) for _class_name, _conf, x1, y1, x2, y2 in boxes]


def occupancy_px(mask: np.ndarray, boxes) -> tuple[int, float]:
    """像素口径的车辆占用率（主指标）：(被覆盖的道路像素数, 占用率)。"""
    road_px = int(mask.sum())
    if road_px == 0:
        return 0, 0.0
    covered = int((boxes_to_mask(mask.shape, _strip(boxes)) & mask).sum())
    return covered, covered / road_px


def occupancy_ground(mask: np.ndarray, boxes, geom: RoadGeometry) -> tuple[float, float]:
    """地面校正口径的占用率（仅作敏感性分析）：(覆盖地面面积, 占用率)。"""
    u = horizon_distance(mask.shape, geom)
    w = np.maximum(u, max(geom.w_min, 1e-6))
    denominator = geom.area_eff or float((mask * (1.0 / w**3)).sum())
    if denominator <= 0:
        return 0.0, 0.0

    covered_ground = 0.0
    for _class_name, _conf, x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        wc = max(
            (geom.a * cx + geom.b * cy + geom.c) / (math.hypot(geom.a, geom.b) or 1.0),
            max(geom.w_min, 1e-6),
        )
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        covered_ground += VEHICLE_ASPECT * area / (wc**2)
    return covered_ground, covered_ground / denominator


def fit_size_law(boxes, geom: RoadGeometry) -> tuple[float, float, int]:
    """拟合 sqrt(检测框面积) = s·u + t，返回 (斜率, R², 样本数)。不要用它反解灭线。"""
    if len(boxes) < 8:
        return 0.0, 0.0, len(boxes)
    norm = math.hypot(geom.a, geom.b) or 1.0
    us, roots = [], []
    for _class_name, _conf, x1, y1, x2, y2 in boxes:
        height = y2 - y1
        if height <= 0:
            continue
        us.append((geom.a * (x1 + x2) / 2.0 + geom.b * (y1 + y2) / 2.0 + geom.c) / norm)
        roots.append(math.sqrt(max(0.0, (x2 - x1) * height)))
    if len(us) < 8:
        return 0.0, 0.0, len(us)
    u_arr = np.asarray(us, dtype=np.float64)
    r_arr = np.asarray(roots, dtype=np.float64)
    slope, intercept = np.polyfit(u_arr, r_arr, 1)
    predicted = slope * u_arr + intercept
    ss_res = float(((r_arr - predicted) ** 2).sum())
    ss_tot = float(((r_arr - r_arr.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r2), len(us)


def w_min_from_slope(slope: float, min_vehicle_px: float = 10.0) -> float:
    """由尺寸律斜率推出检测器尺寸下限（单位：垂直像素）。"""
    if slope <= 1e-9:
        return 1.0
    return max(1.0, min_vehicle_px / slope)


def load_geometry(camera_id: str) -> RoadGeometry | None:
    """读取某相机的几何元数据；不存在或损坏时返回 None。"""
    path = dp.road_mask_meta_path(camera_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    known = {f for f in RoadGeometry.__dataclass_fields__}
    return RoadGeometry(**{k: v for k, v in payload.items() if k in known})


def save_geometry(geom: RoadGeometry) -> Path:
    """写入某相机的几何元数据（保持已有文件的其它键）。"""
    path = dp.road_mask_meta_path(geom.camera_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(asdict(geom))
    payload["horizon_row"] = geom.horizon_row
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_box_map(detections_csv: Path) -> dict[str, list[tuple]]:
    """读取检测 CSV：{image_path: [(class_name, confidence, x1, y1, x2, y2), ...]}。"""
    grouped: dict[str, list[tuple]] = {}
    if not Path(detections_csv).is_file():
        return grouped
    with Path(detections_csv).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            except (KeyError, TypeError, ValueError):
                continue
            try:
                confidence = float(row.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            grouped.setdefault(row.get("image_path", ""), []).append(
                (row.get("class_name", ""), confidence, *box)
            )
    return grouped


def describe(geom: RoadGeometry) -> str:
    """单行摘要，便于日志与调试。"""
    horizon = f"{geom.horizon_row:.0f}" if geom.horizon_row is not None else "n/a(斜)"
    return (
        f"camera {geom.camera_id}: area_frac={geom.area_frac:.3f} "
        f"y_horizon={horizon} w_min={geom.w_min:.0f} lanes={geom.n_lanes} "
        f"size R2={geom.size_r2:.3f} reviewed={geom.reviewed}"
    )
