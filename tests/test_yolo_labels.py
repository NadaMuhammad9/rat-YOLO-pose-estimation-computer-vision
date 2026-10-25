import pandas as pd
from PIL import Image

from rat_yolo_pose_estimation.data.data_prep import KEYPOINT_COLUMNS
from rat_yolo_pose_estimation.data.yolo_labels import (
    csv_to_yolo_labels,
    find_missing_labels,
)


def _annotation(image_name: str, bbox: str = "20_10_180_90", **keypoints: str) -> dict[str, str]:
    return {
        "imageName": image_name,
        "BBox": bbox,
        **{name: keypoints.get(name, "10_10_1") for name in KEYPOINT_COLUMNS},
    }


def _write_image(path, size=(200, 100)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def test_csv_to_yolo_labels_normalizes_bbox_and_keypoints(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_image(images / "train" / "frame.jpg")
    csv_file = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            _annotation(
                r"some\other\frame.jpg",
                **{
                    KEYPOINT_COLUMNS[0]: "40_20_1",
                    KEYPOINT_COLUMNS[1]: "0_0_1",
                    KEYPOINT_COLUMNS[2]: "60_50_0",
                    KEYPOINT_COLUMNS[3]: "bad",
                },
            )
        ]
    ).to_csv(csv_file, index=False)

    result = csv_to_yolo_labels(csv_file, images, labels)

    assert result == {"matched": 1}
    label = labels / "train" / "frame.txt"
    assert label.exists()
    values = label.read_text(encoding="utf-8").strip().split()
    assert values[:5] == ["0", "0.500000", "0.500000", "0.800000", "0.800000"]
    assert len(values) == 5 + len(KEYPOINT_COLUMNS) * 3
    assert values[5:11] == ["0.200000", "0.200000", "1", "0.000000", "0.000000", "0"]
    assert values[11:14] == ["0.300000", "0.500000", "2"]
    assert values[14:17] == ["0.000000", "0.000000", "0"]


def test_csv_to_yolo_labels_skips_unmatched_images_and_invalid_boxes(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_image(images / "present.png")
    csv_file = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            _annotation("missing.jpg"),
            _annotation("present.png", bbox="not-a-box"),
        ]
    ).to_csv(csv_file, index=False)

    result = csv_to_yolo_labels(csv_file, images, labels)

    assert result == {"matched": 0}
    assert not labels.exists()


def test_csv_to_yolo_labels_appends_multiple_annotations_to_same_image(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_image(images / "camera" / "frame.jpeg")
    csv_file = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            _annotation("frame.jpeg", bbox="0_0_100_100"),
            _annotation("frame.jpeg", bbox="50_0_200_100"),
        ]
    ).to_csv(csv_file, index=False)

    result = csv_to_yolo_labels(csv_file, images, labels)

    label_lines = (labels / "camera" / "frame.txt").read_text(encoding="utf-8").splitlines()
    assert result == {"matched": 2}
    assert len(label_lines) == 2
    assert label_lines[0].split()[1:5] == ["0.250000", "0.500000", "0.500000", "1.000000"]
    assert label_lines[1].split()[1:5] == ["0.625000", "0.500000", "0.750000", "1.000000"]


def test_find_missing_labels_returns_only_images_without_relative_label(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_image(images / "train" / "labeled.jpg")
    missing = images / "val" / "missing.PNG"
    _write_image(missing)
    (images / "train" / "notes.txt").write_text("not an image", encoding="utf-8")
    (labels / "train").mkdir(parents=True)
    (labels / "train" / "labeled.txt").write_text("label", encoding="utf-8")

    assert find_missing_labels(images, labels) == [missing]
