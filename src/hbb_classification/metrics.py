"""Threshold-independent and HEP working-point metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true), np.asarray(y_score, dtype=float)


def threshold_grid(y_score: np.ndarray, n_thresholds: int = 200) -> np.ndarray:
    finite = y_score[np.isfinite(y_score)]
    if finite.size == 0:
        return np.array([0.0])
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low == high:
        return np.array([low])
    # Include a point just above the max so the last bin can reject everything.
    return np.linspace(low, high, n_thresholds)


def punzi_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    """Punzi FoM with a=2: epsilon / (1 + sqrt(B)). B=0 yields epsilon, not 0."""
    y_true, y_score = _as_arrays(y_true, y_score)
    selected = y_score > threshold
    n_signal = float(np.sum(y_true == 1))
    if n_signal == 0:
        raise ValueError("Punzi requires at least one signal label.")
    epsilon = float(np.sum(selected & (y_true == 1)) / n_signal)
    n_background = float(np.sum(selected & (y_true == 0)))
    score = epsilon / (1.0 + np.sqrt(n_background))
    return {
        "threshold": float(threshold),
        "signal_efficiency": epsilon,
        "n_background": n_background,
        "punzi": float(score),
    }


def punzi_curve(y_true, y_score, n_thresholds: int = 200) -> dict[str, np.ndarray]:
    y_true, y_score = _as_arrays(y_true, y_score)
    thresholds = threshold_grid(y_score, n_thresholds=n_thresholds)
    punzi = np.array(
        [punzi_at_threshold(y_true, y_score, threshold)["punzi"] for threshold in thresholds]
    )
    return {"thresholds": thresholds, "punzi": punzi}


def best_punzi_threshold(y_true, y_score, n_thresholds: int = 200) -> dict[str, float]:
    curve = punzi_curve(y_true, y_score, n_thresholds=n_thresholds)
    index = int(np.argmax(curve["punzi"]))
    threshold = float(curve["thresholds"][index])
    detail = punzi_at_threshold(y_true, y_score, threshold)
    return detail


def background_rejection_at_efficiency(
    y_true,
    y_score,
    target_efficiency: float,
) -> dict[str, float]:
    """Background rejection 1/FPR at the lowest threshold that reaches the target signal efficiency."""
    y_true, y_score = _as_arrays(y_true, y_score)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_score)
    reachable = np.where(true_positive_rate >= target_efficiency)[0]
    if reachable.size == 0:
        return {
            "target_signal_efficiency": target_efficiency,
            "signal_efficiency": float("nan"),
            "false_positive_rate": float("nan"),
            "background_rejection": float("nan"),
            "threshold": float("nan"),
            "reached": False,
        }
    index = int(reachable[0])
    fpr = float(false_positive_rate[index])
    if fpr <= 0:
        rejection = float("inf")
    else:
        rejection = 1.0 / fpr
    threshold = float(thresholds[index]) if index < len(thresholds) else float("nan")
    return {
        "target_signal_efficiency": target_efficiency,
        "signal_efficiency": float(true_positive_rate[index]),
        "false_positive_rate": fpr,
        "background_rejection": rejection,
        "threshold": threshold,
        "reached": True,
    }


def _class_block(report: dict[str, Any], label: int) -> dict[str, Any]:
    for key in (str(label), str(float(label)), label):
        if key in report:
            block = report[key]
            return {
                "precision": block["precision"],
                "recall": block["recall"],
                "f1": block["f1-score"],
                "support": int(block["support"]),
            }
    raise KeyError(f"classification_report is missing class {label}: {list(report)}")


def classification_at_threshold(y_true, y_score, threshold: float) -> dict[str, Any]:
    y_true, y_score = _as_arrays(y_true, y_score)
    y_pred = (y_score > threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "signal_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "signal_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "signal_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "background": _class_block(report, 0),
        "signal": _class_block(report, 1),
    }


def score_metrics(y_true, y_score) -> dict[str, Any]:
    y_true, y_score = _as_arrays(y_true, y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "rejection_at_50pct_eff": background_rejection_at_efficiency(y_true, y_score, 0.50),
        "rejection_at_80pct_eff": background_rejection_at_efficiency(y_true, y_score, 0.80),
    }


def evaluate_split(
    y_true,
    y_score,
    *,
    punzi_threshold: float,
    default_threshold: float | None = 0.5,
) -> dict[str, Any]:
    payload = score_metrics(y_true, y_score)
    payload["at_punzi_threshold"] = classification_at_threshold(y_true, y_score, punzi_threshold)
    payload["at_punzi_threshold"].update(punzi_at_threshold(y_true, y_score, punzi_threshold))
    if default_threshold is not None:
        payload["at_default_threshold"] = classification_at_threshold(
            y_true, y_score, default_threshold
        )
        payload["at_default_threshold"].update(
            punzi_at_threshold(y_true, y_score, default_threshold)
        )
    return payload
