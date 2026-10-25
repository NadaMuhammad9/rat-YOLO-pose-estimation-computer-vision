from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from rat_yolo_pose_estimation.config import CSV_FILE

from .data_prep import (
    IMAGE_EXTENSIONS,
    KEYPOINT_COLUMNS,
    parse_bbox,
    parse_keypoint,
)

__all__ = ["csv_to_yolo_labels", "find_missing_labels"]


def _clean_image_stem(value: str | Path) -> str:
    cleaned = str(value).replace("\\", "/").strip().strip("/")
    return Path(cleaned.rsplit("/", 1)[-1]).stem


def csv_to_yolo_labels(csv_file: str | Path, images_dir: str | Path, labels_dir: str | Path) -> dict[str, int]:
    """Convert CSV annotations to YOLO pose labels."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    frame = pd.read_csv(csv_file)
    image_index = {
        _clean_image_stem(image.name): image for image in images_dir.rglob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    }
    matched = 0
    created_files: set[Path] = set()
    for _, row in frame.iterrows():
        image_name = _clean_image_stem(row["imageName"])
        image = image_index.get(image_name)
        if image is None:
            continue
        with Image.open(image) as opened:
            width, height = opened.size
        bbox = parse_bbox(row["BBox"])
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        keypoints = []
        for name in KEYPOINT_COLUMNS:
            value = parse_keypoint(row[name]) if name in row else None
            if value is None:
                keypoints.append("0.000000 0.000000 0")
                continue
            raw_x, raw_y, visibility = value
            if raw_x == 0 and raw_y == 0:
                kp_x = kp_y = 0.0
                kp_visibility = 0
            else:
                kp_x = raw_x / width
                kp_y = raw_y / height
                kp_visibility = 2 if visibility == 0 else int(visibility)
            keypoints.append(f"{kp_x:.6f} {kp_y:.6f} {kp_visibility}")
        class_id = 0
        line = f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f} {' '.join(keypoints)}\n"
        
        # It preserves the image's relative path and changes its extension from .jpg to .txt
        label_path = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if label_path not in created_files else "a"
        with label_path.open(mode, encoding="utf-8") as output:
            output.write(line)
        created_files.add(label_path)
        matched += 1

    return {"matched": matched}


# just a final check to see if there are any images that don't have a corresponding label file
def find_missing_labels(images_dir: str | Path, labels_dir: str | Path) -> list[Path]:
    """Return images that do not have a matching YOLO label file."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    return [
        image for image in images_dir.rglob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
        and not (labels_dir / image.relative_to(images_dir).with_suffix(".txt")).exists()
    ]

