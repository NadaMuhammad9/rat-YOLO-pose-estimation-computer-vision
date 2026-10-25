from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

from ..config import CSV_FILE, DATASET_ROOT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
KEYPOINT_COLUMNS = (
    "rRP", "lRP", "rFP", "lFP", "tail_root", "head", "neck", "spine", "tail_middle", "tail_end"
)

'''
Dataset Preparation and Annotation Cleaning Utilities

- Parse and validate bounding boxes and keypoints.
- Find images and check annotation validity.
- Remove invalid annotation rows and their images. 
- Find images without matching annotations in the CSV file. 
- Sort image filenames naturally.
- Split images into train, validation, and test sets (With a specific data split ratio)

Note: Cleanup supports dry-run mode. (deleting only if dry_run is False)
'''


def parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if pd.isna(value):
        return None
    parts = str(value).strip().split("_")
    if len(parts) != 4:
        return None
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def parse_keypoint(value: Any) -> tuple[float, float, float] | None:
    if pd.isna(value):
        return None
    parts = str(value).strip().split("_")
    if len(parts) != 3:
        return None
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def validate_bbox(value: Any, image_width: int, image_height: int) -> tuple[bool, str]:
    bbox = parse_bbox(value)
    if bbox is None:
        return False, f"Invalid BBox format: {value}"
    x1, y1, x2, y2 = bbox
    if min(x1, y1, x2, y2) < 0:
        return False, f"Negative BBox coordinate: {value}"
    if x1 > image_width or x2 > image_width:
        return False, f"BBox exceeds image width ({image_width}): {value}"
    if y1 > image_height or y2 > image_height:
        return False, f"BBox exceeds image height ({image_height}): {value}"
    if x1 >= x2:
        return False, f"x1 must be smaller than x2: {value}"
    if y1 >= y2:
        return False, f"y1 must be smaller than y2: {value}"
    return True, ""


def validate_keypoint(value: Any, image_width: int, image_height: int, name: str) -> tuple[bool, str]:
    keypoint = parse_keypoint(value)
    if keypoint is None:
        return False, f"{name}: invalid format: {value}"
    x, y, visibility = keypoint
    if not 0 <= x <= image_width:
        return False, f"{name}: x={x} is outside [0, {image_width}]"
    if not 0 <= y <= image_height:
        return False, f"{name}: y={y} is outside [0, {image_height}]"
    if visibility not in (0, 1, 2):
        return False, f"{name}: invalid visibility={visibility}"
    return True, ""


def validate_row(row: pd.Series, image_width: int, image_height: int) -> tuple[bool, str]:
    valid, reason = validate_bbox(row["BBox"], image_width, image_height)
    if not valid:
        return False, reason
    for name in KEYPOINT_COLUMNS:
        valid, reason = validate_keypoint(row[name], image_width, image_height, name)
        if not valid:
            return False, reason
    return True, ""


def find_image(base_folder: str | Path, image_name: Any) -> Path | None:
    if pd.isna(image_name) or not str(image_name).strip():
        return None
    relative = Path(str(image_name).strip().replace("\\", "/").lstrip("/"))
    candidate = Path(base_folder) / relative
    return candidate if candidate.is_file() else None


def _inspect_annotations(
    csv_file: str | Path, base_folder: str | Path
) -> tuple[pd.DataFrame, dict[str, int], set[int], set[Path]]:
    frame = pd.read_csv(csv_file, dtype=str)
    required = {"imageName", "BBox", *KEYPOINT_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    invalid_rows: set[int] = set()
    images_to_delete: set[Path] = set()
    missing_images = 0
    for index, row in frame.iterrows():
        image = find_image(base_folder, row["imageName"])
        if image is None:
            missing_images += 1
            invalid_rows.add(index)
            continue
        try:
            with Image.open(image) as opened:
                width, height = opened.size
        except Exception:
            invalid_rows.add(index)
            images_to_delete.add(image.resolve())
            continue
        valid, _ = validate_row(row, width, height)
        if not valid:
            invalid_rows.add(index)
            images_to_delete.add(image.resolve())
    stats = {
        "total_rows": len(frame),
        "valid_rows": len(frame) - len(invalid_rows),
        "invalid_rows": len(invalid_rows),
        "missing_images": missing_images,
    }
    return frame, stats, invalid_rows, images_to_delete


def clean_annotations(
    base_folder: str | Path = DATASET_ROOT,
    csv_file: str | Path = CSV_FILE,
    dry_run: bool = True,
) -> dict[str, int]:
    """Remove invalid annotation rows and their images; preview only by default."""
    frame, stats, invalid_rows, images_to_delete = _inspect_annotations(csv_file, base_folder)
    summary = {
        **stats,
        "rows_to_remove": len(invalid_rows),
        "images_to_delete": len(images_to_delete),
        "rows_removed": 0,
        "images_deleted": 0,
    }
    if dry_run:
        return summary

    summary["images_deleted"] = len(_delete_files(images_to_delete, base_folder, dry_run=False))
    if invalid_rows:
        frame.drop(index=list(invalid_rows)).to_csv(csv_file, index=False)
        summary["rows_removed"] = len(invalid_rows)
    return summary

def unmatched_images(
    base_folder: str | Path = DATASET_ROOT,
    csv_file: str | Path = CSV_FILE,
    dry_run: bool = True,
) -> list[Path]:
    """Return images with missing annotations from the CSV, deleting them only when dry_run is False."""
    base_folder = Path(base_folder)
    frame = pd.read_csv(csv_file, dtype=str)
    names = [str(value).replace("\\", "/").strip().strip("/") for value in frame["imageName"].dropna()]
    unmatched: list[Path] = []
    for folder in sorted(path for path in base_folder.iterdir() if path.is_dir()):
        images = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        for image in images:
            relative_name = image.relative_to(base_folder).as_posix()
            if relative_name not in names:
                unmatched.append(image)
    return _delete_files(unmatched, base_folder, dry_run)

def natural_sort(paths: Iterable[str | Path]) -> list[Path]:
    return sorted(
        (Path(path) for path in paths),
        key=lambda path: [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)],
    )


def split_images(
    dataset_root: str | Path = DATASET_ROOT, 
    output_dir: str | Path = Path("data")
) -> dict[str, int]:
    """Copy Tilt_Indoor and Vertical_Indoor into train/val/test splits."""
    dataset_root = Path(dataset_root)
    output_images = Path(output_dir) / "images"
    split_dirs = {name: output_images / name for name in ("train", "val", "test")}
    for directory in split_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    tilt_files = natural_sort(path for path in (dataset_root / "Tilt_Indoor").iterdir() if path.is_file())
    vertical_files = natural_sort(path for path in (dataset_root / "Vertical_Indoor").iterdir() if path.is_file())
    train_end = math.floor(len(tilt_files) * 0.80)
    test_end = math.floor(len(vertical_files) * 0.50)
    selected = {"train": tilt_files[:train_end], "val": tilt_files[train_end:], "test": vertical_files[:test_end]}
    for split, files in selected.items():
        for source in files:
            shutil.copy2(source, split_dirs[split] / source.name)
    return {split: len(files) for split, files in selected.items()}


def _delete_files(paths: Iterable[str | Path], base_folder: str | Path, dry_run: bool) -> list[Path]:
    base = Path(base_folder).resolve()
    deleted: list[Path] = []
    for path in paths:
        candidate = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if candidate != base and base not in candidate.parents:
            continue
        if candidate.is_file():
            deleted.append(candidate)
            if not dry_run:
                candidate.unlink()
    return deleted
