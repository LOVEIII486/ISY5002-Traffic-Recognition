"""Prepare a local traffic-image dataset for the downstream analysis scripts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_paths as dp

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc


TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}Z)")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_timestamp(filename: str) -> datetime | None:
    match = TIMESTAMP_RE.search(filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(input_dir: Path, output_dir: Path, dataset_tag: str = "") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    rows: list[dict] = []
    invalid_rows: list[dict] = []
    hash_to_paths: defaultdict[str, list[str]] = defaultdict(list)

    for index, path in enumerate(files, start=1):
        relative = path.relative_to(input_dir).as_posix()
        camera_id = path.parent.name
        captured_at = parse_timestamp(path.name)
        row = {
            "image_id": f"{camera_id}_{path.stem}",
            "camera_id": camera_id,
            "image_path": str(path.resolve()),
            "relative_path": relative,
            "file_name": path.name,
            "captured_at": captured_at.isoformat() if captured_at else "",
            "width": "",
            "height": "",
            "file_size": path.stat().st_size,
            "sha256": "",
            "is_valid": False,
            "error": "",
        }
        try:
            file_hash = sha256(path)
            row["sha256"] = file_hash
            hash_to_paths[file_hash].append(relative)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                row["width"], row["height"] = image.size
            if captured_at is None:
                row["error"] = "timestamp_not_found"
            else:
                row["is_valid"] = True
        except Exception as exc:  # keep processing the remaining dataset
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if not row["is_valid"]:
            invalid_rows.append(row)
        if index % 500 == 0:
            print(f"Checked {index}/{len(files)} images")

    duplicate_rows = []
    duplicate_groups = 0
    for file_hash, paths in hash_to_paths.items():
        if len(paths) > 1:
            duplicate_groups += 1
            for duplicate_path in paths[1:]:
                duplicate_rows.append({"sha256": file_hash, "duplicate_of": paths[0], "duplicate_path": duplicate_path})

    valid_rows = [row for row in rows if row["is_valid"]]
    by_camera: dict[str, list[dict]] = defaultdict(list)
    for row in valid_rows:
        by_camera[row["camera_id"]].append(row)

    camera_stats = {}
    for camera_id, camera_rows in sorted(by_camera.items()):
        timestamps = sorted(row["captured_at"] for row in camera_rows)
        deltas = [
            (datetime.fromisoformat(after) - datetime.fromisoformat(before)).total_seconds() / 60
            for before, after in zip(timestamps, timestamps[1:])
        ]
        # 采样间隔随数据集变化（2025-10 那批约 5 分钟，2026-09 那批 10 分钟），
        # 阈值必须跟着数据走：写死 7.5 分钟会把 10 分钟采样的每个正常间隔都记成缺口。
        positive = sorted(delta for delta in deltas if delta > 0)
        expected_minutes = positive[len(positive) // 2] if positive else 5.0
        threshold = expected_minutes * 1.5
        gaps = [
            {"from": before, "to": after, "minutes": delta}
            for before, after, delta in zip(timestamps, timestamps[1:], deltas)
            if delta > threshold
        ]
        camera_stats[camera_id] = {
            "image_count": len(camera_rows),
            "first_capture": timestamps[0] if timestamps else None,
            "last_capture": timestamps[-1] if timestamps else None,
            "expected_interval_minutes": expected_minutes,
            "time_gaps_beyond_1_5x_expected": gaps,
        }

    # Keep the manifest intentionally small so downstream scripts have a stable input.
    write_csv(
        output_dir / "image_manifest.csv",
        [{"camera_id": row["camera_id"], "image_path": row["image_path"]} for row in rows],
        ["camera_id", "image_path"],
    )

    timestamps = sorted(row["captured_at"] for row in valid_rows)
    quality = {
        "dataset_tag": dataset_tag,
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "total_images": len(rows),
        "valid_images": len(valid_rows),
        "invalid_images": len(invalid_rows),
        "camera_count": len(camera_stats),
        "images_by_camera": dict(Counter(row["camera_id"] for row in valid_rows)),
        "duplicate_groups": duplicate_groups,
        "duplicate_files": len(duplicate_rows),
        "time_range": {"start": timestamps[0] if timestamps else None, "end": timestamps[-1] if timestamps else None},
    }
    (output_dir / "dataset_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a traffic image dataset for downstream analysis")
    dp.add_tag_argument(parser)
    parser.add_argument("--input", type=Path, default=None, help="Dataset root (default: from the tag registry)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: results/<tag>/02_dataset)")
    args = parser.parse_args()
    tag = dp.resolve_tag(args.dataset_tag)
    input_dir = args.input if args.input is not None else dp.raw_dir(tag)
    output_dir = args.output if args.output is not None else dp.stage_dir(tag, "02_dataset")
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    print(f"Dataset tag: {tag}")
    print(f"Input directory: {input_dir}")
    quality = prepare_dataset(input_dir, output_dir, tag)
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
