from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from .model.metrics import extract_metrics
from .model.compare import compare_validation_and_test

from .config import DEFAULT_DATA_CONFIG

from .model.train import train_model
from .model.validate import validate_model

__all__ = ["main"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, validate, and compare the YOLO pose estimation model."
    )

    parser.add_argument(
        "command",
        choices=("train", "validate", "compare"),
        help="Operation to perform.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Path to the YOLO dataset configuration file.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the model.",
    )

    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to use during validation. Default: val.",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            args.config,
            model_path=str(args.model),
        )
        return

    if args.command == "validate":
        metrics = validate_model(
            args.model,
            args.config,
            args.split,
        )
        pprint(extract_metrics(metrics))
        return

    if args.command == "compare":
        results = compare_validation_and_test(
            model_path=args.model,
            data_config=args.config,
        )
        pprint(results)
        return


if __name__ == "__main__":
    main()