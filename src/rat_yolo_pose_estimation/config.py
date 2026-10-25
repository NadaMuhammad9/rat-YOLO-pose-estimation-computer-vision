from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

# paths 

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DEFAULT_DATA_CONFIG: Final[Path] = PROJECT_ROOT / "config.yaml"
MODEL_PATH: Final[Path] = PROJECT_ROOT / "yolo26s-pose.pt"

# that model has already been trained in google colab
DEFAULT_PREDICTION_MODEL_PATH: Final[Path] = PROJECT_ROOT / "runs/pose/train/weights/best.pt"

DATASET_ROOT = Path(r"C:\Users\nada\Downloads\Lawn scenarios")
CSV_FILE = DATASET_ROOT / "Lawn_scenarios.csv"

@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    train: Path
    val: Path
    test: Path
    keypoint_names: tuple[str, ...]


def _resolve_dataset_root(config_path: Path, values: dict[str, Any]) -> Path:
    root = Path(values.get("path", "."))
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def resolve_split_paths(data_config: str | Path, split: str) -> tuple[Path, Path]:
    """Resolve the image and label directories for one dataset split."""
    config_path = Path(data_config).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        values: dict[str, Any] = yaml.safe_load(file) or {}

    dataset_root = _resolve_dataset_root(config_path, values)
    image_dir = Path(values[split])
    if not image_dir.is_absolute():
        image_dir = dataset_root / image_dir

    parts = list(image_dir.parts)
    if "images" in parts:
        images_index = len(parts) - 1 - parts[::-1].index("images")
        label_parts = parts.copy()
        label_parts[images_index] = "labels"
        label_dir = Path(*label_parts)
    else:
        label_dir = image_dir.parent / "labels" / image_dir.name

    return image_dir.resolve(), label_dir.resolve()


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load dataset paths and keypoint names from a YAML configuration file."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as file:
        values: dict[str, Any] = yaml.safe_load(file) or {}

    root = _resolve_dataset_root(config_path, values)

    def split_path(name: str) -> Path:
        value = Path(values[name])
        return value if value.is_absolute() else root / value

    return DatasetConfig(
        root=root,
        train=split_path("train"),
        val=split_path("val"),
        test=split_path("test"),
        keypoint_names=tuple(values.get("kpt_names", ())),
    )
