from __future__ import annotations

from pathlib import Path
from typing import Any

from .metrics import extract_metrics
from ultralytics import YOLO

__all__ = ["validate_model"]


def validate_model(model_path: str | Path , data_config: str | Path, split: str | None = None) -> Any:

    model = YOLO(model_path)
    arguments = {"data": str(data_config)}
    if split is not None:
        arguments["split"] = split

    return model.val(**arguments)