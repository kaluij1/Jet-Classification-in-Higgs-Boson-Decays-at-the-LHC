"""Shared synthetic cms_Hbb-shaped fixtures. CI never downloads the 105 MB CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hbb_classification.data import EXPECTED_COLUMNS, FEATURE_COLUMNS, SENTINEL_COLUMNS


def make_toy_frame(
    n_signal: int = 30,
    n_background: int = 30,
    *,
    seed: int = 0,
    n_sentinel: int = 2,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_rows = n_signal + n_background
    data = {"Unnamed: 0": np.arange(n_rows, dtype=int)}
    for column in FEATURE_COLUMNS:
        if column == "nSV":
            values = rng.integers(0, 5, size=n_rows).astype(float)
        elif column == "jetNTracks":
            values = rng.integers(5, 40, size=n_rows).astype(float)
        else:
            values = rng.normal(loc=1.0, scale=0.4, size=n_rows)
        # Nudge a few physics scores so signal is slightly separable.
        if column in {"trackSipdSig_0", "trackSip2dSigAboveBottom_0", "tau_flightDistance2dSig_0"}:
            values[:n_signal] += 1.2
        data[column] = values

    labels = np.concatenate([np.ones(n_signal), np.zeros(n_background)])
    data["isBackground"] = 1.0 - labels
    data["isSignal"] = labels

    frame = pd.DataFrame(data)[EXPECTED_COLUMNS]
    if n_sentinel:
        for index in range(n_sentinel):
            for column in SENTINEL_COLUMNS:
                frame.loc[index, column] = -1.0
    return frame


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    return make_toy_frame()
