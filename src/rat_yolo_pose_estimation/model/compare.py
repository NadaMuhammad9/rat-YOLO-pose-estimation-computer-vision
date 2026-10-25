from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from ..config import resolve_split_paths
from .metrics import calculate_pck, extract_metrics
from .validate import validate_model


def compare_validation_and_test(
    model_path: str | Path,
    data_config: str | Path,
    pck_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, dict[str, Any]]:
    """Evaluate a pose model on validation and test splits."""

    with open(data_config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    keypoint_count = len(config["kpt_names"])

    results: dict[str, dict[str, Any]] = {}

    # Load the model once and reuse it for PCK predictions.
    model = YOLO(model_path)

    for split in ("val", "test"):

        metrics = validate_model(
            model_path=model_path,
            data_config=data_config,
            split=split,
        )

        image_dir, label_dir = resolve_split_paths(
            data_config=data_config,
            split=split,
        )

        pck_results = calculate_pck(
            model=model,
            image_dir=image_dir,
            label_dir=label_dir,
            keypoint_count=keypoint_count,
            thresholds=pck_thresholds,
        )

        extracted = extract_metrics(metrics)

        kpt_names = config["kpt_names"]

        extracted["pck"] = {
            threshold: dict(zip(kpt_names, pck.tolist()))
            for threshold, pck in pck_results.items()
            }

        extracted["pck_mean"] = {
            threshold: float(pck.mean())
            for threshold, pck in pck_results.items()
        }

        results[split] = extracted

    return results
