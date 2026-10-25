from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from rat_yolo_pose_estimation.data.data_prep import (
    KEYPOINT_COLUMNS,
    clean_annotations,
    find_image,
    natural_sort,
    parse_bbox,
    parse_keypoint,
    split_images,
    unmatched_images,
    validate_bbox,
    validate_keypoint,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1_2_30_40", (1.0, 2.0, 30.0, 40.0)),
        (None, None),
        ("1_2_3", None),
        ("left_2_30_40", None),
    ],
)
def test_parse_bbox(value, expected):
    assert parse_bbox(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.5_2_1", (1.5, 2.0, 1.0)),
        (None, None),
        ("1_2", None),
        ("1_two_1", None),
    ],
)
def test_parse_keypoint(value, expected):
    assert parse_keypoint(value) == expected


def test_validate_bbox_accepts_coordinates_on_image_edges():
    assert validate_bbox("0_0_100_50", 100, 50) == (True, "")


@pytest.mark.parametrize(
    ("value", "image_width", "image_height"),
    [
        ("invalid", 100, 50),
        ("-1_0_20_20", 100, 50),
        ("0_0_101_20", 100, 50),
        ("0_51_20_60", 100, 50),
        ("20_0_20_20", 100, 50),
        ("0_20_20_20", 100, 50),
    ],
)
def test_validate_bbox_rejects_invalid_boxes(value, image_width, image_height):
    valid, reason = validate_bbox(value, image_width, image_height)

    assert not valid
    assert reason


@pytest.mark.parametrize(
    ("value", "expected_valid"),
    [
        ("0_0_0", True),
        ("100_50_2", True),
        ("101_0_1", False),
        ("0_-1_1", False),
        ("0_0_3", False),
        ("invalid", False),
    ],
)
def test_validate_keypoint_checks_bounds_visibility_and_format(value, expected_valid):
    valid, reason = validate_keypoint(value, 100, 50, "head")

    assert valid is expected_valid
    assert (reason == "") is expected_valid


def test_find_image_normalizes_windows_separators_and_rejects_missing_names(tmp_path):
    image = tmp_path / "camera" / "frame.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")

    assert find_image(tmp_path, r"camera\frame.jpg") == image
    assert find_image(tmp_path, "  ") is None
    assert find_image(tmp_path, "camera/missing.jpg") is None


def _valid_annotation(image_name: str, bbox: str = "1_1_9_9") -> dict[str, str]:
    return {
        "imageName": image_name,
        "BBox": bbox,
        **{name: "5_5_1" for name in KEYPOINT_COLUMNS},
    }


def test_clean_annotations_dry_run_reports_invalid_rows_without_changes(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 10)).save(images / "image_with_valid_annotation.jpg")
    Image.new("RGB", (10, 10)).save(images / "image_with_invalid_annotation.jpg")
    csv_file = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            _valid_annotation("image_with_valid_annotation.jpg"),
            _valid_annotation("image_with_invalid_annotation.jpg", "8_8_2_2"),
            _valid_annotation("missing.jpg"),
        ]
    ).to_csv(csv_file, index=False)

    result = clean_annotations(images, csv_file)

    assert result == {
        "total_rows": 3,
        "valid_rows": 1,
        "invalid_rows": 2,
        "missing_images": 1,
        "rows_to_remove": 2,
        "images_to_delete": 1,
        "rows_removed": 0,
        "images_deleted": 0,
    }
    assert (images / "image_with_invalid_annotation.jpg").exists()
    assert len(pd.read_csv(csv_file)) == 3


def test_clean_annotations_removes_invalid_rows_and_their_images(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 10)).save(images / "image_with_valid_annotation.jpg")
    Image.new("RGB", (10, 10)).save(images / "image_with_invalid_annotation.jpg")
    csv_file = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            _valid_annotation("image_with_valid_annotation.jpg"),
            _valid_annotation("image_with_invalid_annotation.jpg", "8_8_2_2"),
        ]
    ).to_csv(csv_file, index=False)

    result = clean_annotations(images, csv_file, dry_run=False)

    assert result["rows_removed"] == 1
    assert result["images_deleted"] == 1
    assert (images / "image_with_valid_annotation.jpg").exists()
    assert not (images / "image_with_invalid_annotation.jpg").exists()
    assert pd.read_csv(csv_file)["imageName"].tolist() == ["image_with_valid_annotation.jpg"]


def test_unmatched_images_previews_then_deletes_only_unmatched_images(tmp_path):
    dataset = tmp_path / "dataset"
    camera = dataset / "camera"
    camera.mkdir(parents=True)
    matched = camera / "matched.jpg"
    unmatched = camera / "unmatched.PNG"
    ignored = camera / "notes.txt"
    matched.write_bytes(b"matched")
    unmatched.write_bytes(b"unmatched")
    ignored.write_text("not an image", encoding="utf-8")
    csv_file = tmp_path / "annotations.csv"
    pd.DataFrame({"imageName": ["camera/matched.jpg"]}).to_csv(csv_file, index=False)

    assert unmatched_images(dataset, csv_file) == [unmatched.resolve()]
    assert unmatched.exists()
    assert unmatched_images(dataset, csv_file, dry_run=False) == [unmatched.resolve()]
    assert matched.exists()
    assert not unmatched.exists()
    assert ignored.exists()


def test_natural_sort_orders_numeric_filename_parts_numerically():
    paths = [Path("frame10.jpg"), Path("frame2.jpg"), Path("frame1.jpg")]

    assert natural_sort(paths) == [
        Path("frame1.jpg"),
        Path("frame2.jpg"),
        Path("frame10.jpg"),
    ]


def test_split_images_creates_expected_natural_ordered_splits(tmp_path):
    dataset = tmp_path / "dataset"
    subfolder1 = dataset / "Tilt_Indoor"
    subfolder2 = dataset / "Vertical_Indoor"
    subfolder1.mkdir(parents=True)
    subfolder2.mkdir()
    for name in ("frame10.jpg", "frame2.jpg", "frame1.jpg", "frame3.jpg", "frame4.jpg"):
        (subfolder1 / name).write_text(name, encoding="utf-8")
    for name in ("image4.jpg", "image2.jpg", "image1.jpg", "image3.jpg"):
        (subfolder2 / name).write_text(name, encoding="utf-8")

    output = tmp_path / "output"
    result = split_images(dataset, output)

    assert result == {"train": 4, "val": 1, "test": 2}
    assert {path.name for path in (output / "images" / "train").iterdir()} == {
        "frame1.jpg",
        "frame2.jpg",
        "frame3.jpg",
        "frame4.jpg",
    }
    assert [path.name for path in (output / "images" / "val").iterdir()] == ["frame10.jpg"]
    assert sorted(path.name for path in (output / "images" / "test").iterdir()) == [
        "image1.jpg",
        "image2.jpg",
    ]
    assert (output / "images" / "val" / "frame10.jpg").read_text(encoding="utf-8") == "frame10.jpg"
