import numpy as np
import pytest

from hbb_classification.metrics import punzi_at_threshold


def test_punzi_known_values() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.05, 0.04, 0.03])
    result = punzi_at_threshold(y_true, y_score, 0.5)
    assert result["signal_efficiency"] == 0.5
    assert result["n_background"] == 1.0
    assert result["punzi"] == pytest.approx(0.5 / (1.0 + 1.0))


def test_punzi_zero_background_is_efficiency() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.1, 0.2])
    result = punzi_at_threshold(y_true, y_score, 0.5)
    assert result["n_background"] == 0.0
    assert result["signal_efficiency"] == 1.0
    assert result["punzi"] == 1.0
