from __future__ import annotations

from pathlib import Path

from ..config import DEFAULT_DATA_CONFIG, MODEL_PATH

__all__ = ["train_model"]


def train_model(
    data_config: str | Path = DEFAULT_DATA_CONFIG,
    model_path: str | Path = MODEL_PATH,
    **overrides,
):
    """Train a YOLO pose model using the project config."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_config),
        imgsz=1024,
        epochs=50,
        patience=15,
        batch=16,
        workers=8,
        degrees=10.0,
        perspective=0.001,
        shear=5.0,
        translate=0.03,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.1,
        hsv_v=0.2,
        close_mosaic=10,
        **overrides,
    )
    return results
