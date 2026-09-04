"""Stratified train / validation / test split on the raw class prior."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from hbb_classification.data import feature_matrix

TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20
TEST_FRACTION = 0.20


def _class_counts(labels: pd.Series) -> dict[str, int]:
    return {
        "background": int((labels == 0).sum()),
        "signal": int((labels == 1).sum()),
    }


def make_feature_label_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = feature_matrix(frame).reset_index(drop=True)
    labels = frame["isSignal"].astype(int).reset_index(drop=True)
    labels.name = "isSignal"
    return features, labels


def stratified_train_val_test(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    random_state: int = 42,
) -> dict[str, Any]:
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must sum to 1.")

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        labels,
        test_size=1.0 - train_fraction,
        random_state=random_state,
        stratify=labels,
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=relative_test,
        random_state=random_state,
        stratify=y_holdout,
    )

    return {
        "x_train": x_train.reset_index(drop=True),
        "x_val": x_val.reset_index(drop=True),
        "x_test": x_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "summary": {
            "fractions": {
                "train": train_fraction,
                "val": val_fraction,
                "test": test_fraction,
            },
            "n_train": int(len(x_train)),
            "n_val": int(len(x_val)),
            "n_test": int(len(x_test)),
            "class_counts": {
                "train": _class_counts(y_train),
                "val": _class_counts(y_val),
                "test": _class_counts(y_test),
            },
        },
    }
