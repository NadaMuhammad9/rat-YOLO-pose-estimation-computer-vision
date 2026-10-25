from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


__all__ = ["calculate_pck", "extract_metrics"]

def calculate_pck(
    model: Any,
    image_dir: str | Path,
    label_dir: str | Path,
    keypoint_count: int,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[float, np.ndarray]:
    """Calculate Per-keypoint PCK at multiple thresholds."""
    correct = {
        threshold: np.zeros(keypoint_count, dtype=int)
        for threshold in thresholds
    }

    total = np.zeros(keypoint_count, dtype=int)

    image_paths = [
        image
        for image in Path(image_dir).iterdir()
        if image.is_file()
        and image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    for image_path in image_paths:

        label_path = Path(label_dir) / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        results = model.predict(
            source=str(image_path),
            verbose=False
        )

        if (
            not results
            or results[0].keypoints is None
            or len(results[0].keypoints.xy) == 0
        ):
            continue

        predicted = results[0].keypoints.xy.cpu().numpy()
        height, width = results[0].orig_shape

        try:
            labels = np.loadtxt(label_path, ndmin=2)
        except Exception:
            continue

        for label in labels:

            x_center, y_center, box_width, box_height = label[1:5]
            values = label[5:]

            if len(values) == keypoint_count * 3:
                keypoints = values.reshape(keypoint_count, 3)
                visibility = keypoints[:, 2]

            elif len(values) == keypoint_count * 2:
                keypoints = values.reshape(keypoint_count, 2)
                visibility = np.ones(keypoint_count)

            else:
                continue

            ground_truth = keypoints[:, :2] * (width, height)

            if len(predicted) == 1:
                prediction = predicted[0]

            else:
                center = np.array([
                    x_center * width,
                    y_center * height
                ])

                prediction = predicted[
                    np.linalg.norm(
                        predicted.mean(axis=1) - center,
                        axis=1
                    ).argmin()
                ]

            diagonal = np.hypot(
                box_width * width,
                box_height * height
            )

            errors = (
                np.linalg.norm(
                    prediction - ground_truth,
                    axis=1
                ) / (diagonal + 1e-9)
            )

            visible = visibility > 0

            total[visible] += 1

            for threshold in thresholds:
                correct[threshold][visible] += (
                    errors[visible] <= threshold
                )

    return {
        threshold: np.divide(
            correct[threshold],
            total,
            out=np.zeros(keypoint_count, dtype=float),
            where=total != 0
        )
        for threshold in thresholds
    }

def extract_metrics(metrics: Any) -> dict[str, float]:
    """ box, pose, and latency."""
    return {
        "box_precision": metrics.box.mp,
        "box_recall": metrics.box.mr,
        "box_map50": metrics.box.map50,
        "box_map75": metrics.box.map75,
        "box_map50_95": metrics.box.map,
        "pose_precision": metrics.pose.mp,
        "pose_recall": metrics.pose.mr,
        "pose_map50": metrics.pose.map50,
        "pose_map75": metrics.pose.map75,
        "pose_map50_95": metrics.pose.map,
        "preprocess_ms": metrics.speed["preprocess"],
        "inference_ms": metrics.speed["inference"],
        "postprocess_ms": metrics.speed["postprocess"],
    }