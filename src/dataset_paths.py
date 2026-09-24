"""按数据集标签隔离 results 输出：标签解析、路径解析与数据集登记表。"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
TAG_ENV_VAR = "ISY5002_DATASET_TAG"

# 共享目录：不参与标签隔离
SHARED_STAGE_DIRS = {"00_lta_traffic_images"}

# 标签必须以字母或数字开头（允许前置若干下划线，供 `_smoke` 这类临时标签使用），
# 其后只允许字母数字、点、横线、下划线。要求首字符是字母数字，既挡住
# `--dataset-tag ../../x` 逃出 results/，也避免 argparse 把 `-x` 当成选项。
TAG_PATTERN = re.compile(r"_*[A-Za-z0-9][A-Za-z0-9._-]*")

# 数据集登记表。**新增数据集时在这里加一行**：
#   raw_dir  阶段 02 的默认输入目录
#   exp_base 阶段 06 供 LibCity 用的 exp_id 基数（各数据集必须不同，否则会共用评估缓存）
REGISTRY: dict[str, dict[str, object]] = {
    "2025-10-week": {"raw_dir": "datasets/raw-old", "exp_base": 7000},
    "2026-week1": {"raw_dir": "datasets/raw-2026-week1", "exp_base": 7100},
    "2026-week2": {"raw_dir": "datasets/raw-2026-week2", "exp_base": 7200},
}


def _registry_entry(tag: str) -> dict[str, object]:
    """取登记信息；未登记时给出可直接照抄的报错。"""
    entry = REGISTRY.get(tag)
    if entry is None:
        suggestion = 7200 + 100 * len(REGISTRY)
        raise SystemExit(
            f"Dataset tag {tag!r} is not registered in REGISTRY (src/dataset_paths.py).\n"
            f"Add a line there, for example:\n"
            f'    "{tag}": {{"raw_dir": "datasets/raw-{tag}", "exp_base": {suggestion}}},'
        )
    return entry


def _validate(tag: str) -> str:
    """校验标签可安全用作单层目录名。"""
    if not TAG_PATTERN.fullmatch(tag):
        raise SystemExit(
            f"Invalid dataset tag {tag!r}: only letters, digits, '.', '-' and '_' are allowed, "
            "and it must start with a letter or digit (leading underscores are allowed)."
        )
    return tag


def existing_tags() -> list[str]:
    """列出 results/ 下已存在的数据集标签。"""
    if not RESULTS_ROOT.is_dir():
        return []
    tags = []
    for entry in sorted(RESULTS_ROOT.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_") or entry.name in SHARED_STAGE_DIRS:
            continue
        try:
            has_stage = any(child.is_dir() and re.match(r"\d{2}_", child.name) for child in entry.iterdir())
        except OSError:
            continue
        if has_stage:
            tags.append(entry.name)
    return tags


def resolve_tag(explicit: str | None = None, interactive: bool = False) -> str:
    """确定本次运行的数据集标签。

    顺序：显式参数 → 环境变量 → 唯一标签时自动采用 → （interactive 且是终端时）让用户选 → 报错。

    默认**不猜**：会写数据的阶段（02-06b）猜错会把结果静默写进另一个数据集的目录，
    而 results/ 不受 git 保护、无法回滚。只读的展示/评估阶段（07/08/09）传
    interactive=True，在终端里给出选择——它们只读已算好的结果，猜错的代价小得多。
    """
    if explicit:
        return _validate(explicit)
    from_env = os.environ.get(TAG_ENV_VAR)
    if from_env:
        return _validate(from_env)
    tags = existing_tags()
    if len(tags) == 1:
        print(f"[dataset_paths] 未指定标签，只有一个，自动采用：{tags[0]}")
        return tags[0]

    if interactive and sys.stdin is not None and sys.stdin.isatty() and tags:
        print("[dataset_paths] 未指定 --dataset-tag，可用的数据集标签：")
        for index, tag in enumerate(tags, start=1):
            print(f"    {index}) {tag}")
        try:
            answer = input(f"选择序号 [默认 {len(tags)} = {tags[-1]}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n已取消。") from None
        if not answer:
            print(f"[dataset_paths] 采用 {tags[-1]}")
            return tags[-1]
        if answer.isdigit() and 1 <= int(answer) <= len(tags):
            return tags[int(answer) - 1]
        return _validate(answer)

    script = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "<脚本名>"
    if not tags:
        raise SystemExit(
            "没有指定数据集标签，且 results/ 下还没有任何标签目录。\n"
            f"  先跑阶段 02 生成一个，例如：\n"
            f"    python src/02_prepare_dataset.py --dataset-tag <tag>\n"
            f"  已登记的标签：{', '.join(REGISTRY) or '(none)'}"
        )
    raise SystemExit(
        f"没有指定数据集标签，而 results/ 下有 {len(tags)} 个，无法自动判断：\n"
        + "\n".join(f"    - {tag}" for tag in tags)
        + f"\n\n二选一：\n"
        + f"    python src/{script} --dataset-tag {tags[-1]}\n"
        + f"  或设一次环境变量：{TAG_ENV_VAR}={tags[-1]}"
    )


def stage_dir(tag: str, stage: str) -> Path:
    """某数据集某阶段的输出目录（纯函数，不建目录）。"""
    return RESULTS_ROOT / tag / stage


def shared_dir(stage: str) -> Path:
    """共享目录（相机参考元数据），不参与标签隔离。"""
    return RESULTS_ROOT / stage


def exp_base(tag: str) -> int:
    """阶段 06 用于 LibCity exp_id 的基数（各数据集必须不同，否则会共用评估缓存）。"""
    return int(_registry_entry(tag)["exp_base"])


def raw_dir(tag: str) -> Path:
    """阶段 02 的默认输入目录。"""
    return PROJECT_ROOT / str(_registry_entry(tag)["raw_dir"])


def road_mask_dir() -> Path:
    """道路掩码目录（相机级共享资产，纳入版本控制）。"""
    return PROJECT_ROOT / "road_masks"


def road_mask_path(camera_id: str) -> Path:
    """某相机的二值道路掩码（1920x1080 PNG，255=道路）。"""
    return road_mask_dir() / f"{camera_id}.png"


def road_mask_meta_path(camera_id: str) -> Path:
    """某相机掩码的元数据（方法、面积、灭线、车道数、是否已人工确认）。"""
    return road_mask_dir() / f"{camera_id}.json"


def clean_plate_path(camera_id: str) -> Path:
    """某相机的时间中值底图（车辆被中值抹掉，用于底图残差信号）。"""
    return road_mask_dir() / f"{camera_id}_cleanplate.jpg"


def resolve_stored_path(tag: str, camera_id: str, stored: str | Path) -> Path:
    """把 CSV 里记录的图片路径还原成本机可用路径（旧数据集存的是另一台机器的绝对路径）。"""
    path = Path(stored)
    if path.is_file():
        return path
    candidate = raw_dir(tag) / str(camera_id) / path.name
    if candidate.is_file():
        return candidate
    return path


def add_tag_argument(parser) -> None:
    """给 argparse 加上统一的 ``--dataset-tag``。"""
    parser.add_argument(
        "--dataset-tag",
        default=None,
        help=f"Dataset tag used to namespace results output (defaults to ${TAG_ENV_VAR})",
    )
