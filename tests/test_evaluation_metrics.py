from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rat_yolo_pose_estimation.model.metrics import calculate_pck, extract_metrics


class FakeKeypoints:
    def __init__(self, coordinates: np.ndarray):
        self.xy = FakeTensor(coordinates)


class FakeTensor:
    def __init__(self, value: np.ndarray):
        self.value = value

    def __len__(self):
        return len(self.value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeModel:
    def __init__(self, coordinates: np.ndarray | None):
        self.coordinates = coordinates

    def predict(self, source: str, verbose: bool):
        if self.coordinates is None:
            return []

        return [
            SimpleNamespace(
                keypoints=FakeKeypoints(self.coordinates),
                orig_shape=(100, 200),
            )
        ]


def write_sample_files(tmp_path: Path, label: str):
    images = tmp_path / "images"
    labels = tmp_path / "labels"

    images.mkdir()
    labels.mkdir()

    (images / "frame.jpg").write_bytes(b"image")
    (labels / "frame.txt").write_text(label, encoding="utf-8")

    return images, labels


def test_calculate_pck_counts_visible_keypoints_only(tmp_path: Path):
    images, labels = write_sample_files(
        tmp_path,
        "0 0.5 0.5 0.5 0.5 0.25 0.5 1 0.75 0.5 0 0.5 0.5 1",
    )

    predictions = np.array([
        [[50, 50], [150, 50], [100, 50]]
    ])

    result = calculate_pck(
        FakeModel(predictions),
        images,
        labels,
        keypoint_count=3,
    )

    np.testing.assert_array_equal(
        result[0.05],
        [1.0, 0.0, 1.0],
    )

    np.testing.assert_array_equal(
        result[0.10],
        [1.0, 0.0, 1.0],
    )

    np.testing.assert_array_equal(
        result[0.20],
        [1.0, 0.0, 1.0],
    )


def test_calculate_pck_accepts_two_value_keypoints_and_threshold(tmp_path: Path):
    images, labels = write_sample_files(
        tmp_path,
        "0 0.5 0.5 0.5 0.5 0.25 0.5 0.75 0.5",
    )

    predictions = np.array([
        [[52, 50], [150, 50]]
    ])

    result = calculate_pck(
        FakeModel(predictions),
        images,
        labels,
        keypoint_count=2,
        thresholds=(0.05,),
    )

    np.testing.assert_array_equal(
        result[0.05],
        [1.0, 1.0],
    )


def test_calculate_pck_selects_prediction_nearest_label_center(tmp_path: Path):
    images, labels = write_sample_files(
        tmp_path,
        "0 0.75 0.5 0.2 0.5 0.75 0.5",
    )

    predictions = np.array([
        [[50, 50]],
        [[150, 50]],
    ])

    result = calculate_pck(
        FakeModel(predictions),
        images,
        labels,
        keypoint_count=1,
    )

    np.testing.assert_array_equal(
        result[0.05],
        [1.0],
    )


def test_calculate_pck_returns_zero_when_predictions_are_empty(tmp_path: Path):
    images, labels = write_sample_files(
        tmp_path,
        "0 0.5 0.5 0.5 0.5 0.5 0.5",
    )

    result = calculate_pck(
        FakeModel(None),
        images,
        labels,
        keypoint_count=1,
    )

    np.testing.assert_array_equal(
        result[0.05],
        [0.0],
    )


def test_extract_metrics():
    metrics = SimpleNamespace(
        box=SimpleNamespace(mp=0.1, mr=0.2, map50=0.3,map75=0.4, map=0.5),
        pose=SimpleNamespace(mp=0.6, mr=0.7, map50=0.8, map75=0.9, map=1.0),
        speed={
            "preprocess": 1.0,
            "inference": 2.0,
            "postprocess": 3.0,
        },
    )

    result = extract_metrics(metrics)

    assert result["box_map50_95"] == 0.5
    assert result["pose_map75"] == 0.9
    assert result["inference_ms"] == 2.0

