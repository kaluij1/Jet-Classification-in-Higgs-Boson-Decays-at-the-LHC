"""Train-only sentinel handling for HEP placeholder values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from hbb_classification.data import SENTINEL_COLUMNS, SENTINEL_VALUE


class SentinelHandler(BaseEstimator, TransformerMixin):
    """Add missingness indicators and median-impute sentinel values.

    Medians are computed on non-sentinel training values only. Call ``fit``
    on the training split and ``transform`` on every later split.
    """

    def __init__(
        self,
        columns: tuple[str, ...] = SENTINEL_COLUMNS,
        sentinel: float = SENTINEL_VALUE,
    ) -> None:
        self.columns = columns
        self.sentinel = sentinel

    def fit(self, X: pd.DataFrame, y=None):
        frame = self._as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        self.medians_: dict[str, float] = {}
        self.n_sentinel_in_fit_: dict[str, int] = {}
        for column in self.columns:
            if column not in frame.columns:
                raise ValueError(f"Sentinel column {column!r} is not in X.")
            values = frame[column].to_numpy(dtype=float)
            mask = values == self.sentinel
            self.n_sentinel_in_fit_[column] = int(mask.sum())
            valid = values[~mask]
            if valid.size == 0:
                raise ValueError(f"No non-sentinel values in {column!r} to impute from.")
            self.medians_[column] = float(np.median(valid))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._as_frame(X).copy()
        for column in self.columns:
            indicator = self.indicator_name(column)
            mask = frame[column].to_numpy(dtype=float) == self.sentinel
            frame[indicator] = mask.astype(np.float64)
            frame.loc[mask, column] = self.medians_[column]
        return frame

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(input_features) if input_features is not None else list(self.feature_names_in_)
        names.extend(self.indicator_name(column) for column in self.columns)
        return np.asarray(names, dtype=object)

    @staticmethod
    def indicator_name(column: str) -> str:
        return f"{column}_is_sentinel"

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("SentinelHandler expects a pandas DataFrame so column names are preserved.")
