"""Copy cms_Hbb.csv into data/raw/ and verify size plus SHA-256."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from hbb_classification.data import (
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    default_raw_path,
    validate_file,
)


def prepare_dataset(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    destination = destination.resolve()

    if source != destination:
        shutil.copy2(source, destination)

    validate_file(destination, check_hash=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy cms_Hbb.csv into data/raw/ and verify the checksum."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to an existing cms_Hbb.csv",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=default_raw_path(),
        help="Destination path (default: data/raw/cms_Hbb.csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = prepare_dataset(args.source, args.destination)
    print(f"Dataset ready at {path}")
    print(f"Size:    {EXPECTED_SIZE_BYTES:,} bytes")
    print(f"SHA-256: {EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
