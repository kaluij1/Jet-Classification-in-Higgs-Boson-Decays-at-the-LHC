from pathlib import Path

import pandas as pd
import pytest

from hbb_classification.data import (
    EXPECTED_COLUMNS,
    DataValidationError,
    load_table,
    validate_columns,
    validate_labels,
    validate_schema,
)


def test_toy_frame_matches_schema(toy_frame: pd.DataFrame) -> None:
    validate_columns(toy_frame)
    validate_labels(toy_frame)
    validate_schema(toy_frame, require_expected_rows=False)
    assert list(toy_frame.columns) == EXPECTED_COLUMNS


def test_validate_labels_rejects_overlap(toy_frame: pd.DataFrame) -> None:
    bad = toy_frame.copy()
    bad.loc[0, "isSignal"] = 1
    bad.loc[0, "isBackground"] = 1
    with pytest.raises(DataValidationError, match="overlap|complementary"):
        validate_labels(bad)


def test_validate_columns_rejects_missing_feature(toy_frame: pd.DataFrame) -> None:
    bad = toy_frame.drop(columns=["nSV"])
    with pytest.raises(DataValidationError, match="columns"):
        validate_columns(bad)


def test_load_table_skips_row_count(tmp_path: Path, toy_frame: pd.DataFrame) -> None:
    path = tmp_path / "tiny.csv"
    toy_frame.to_csv(path, index=False)
    loaded = load_table(path)
    assert len(loaded) == len(toy_frame)


def test_full_schema_rejects_tiny_row_count(toy_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="rows"):
        validate_schema(toy_frame, require_expected_rows=True)
