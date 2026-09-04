"""Leakage-controlled P1 training and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hbb_classification.data import PHYSICS_SCORE_CANDIDATES, SENTINEL_COLUMNS, SENTINEL_VALUE
from hbb_classification.metrics import best_punzi_threshold, evaluate_split, score_metrics
from hbb_classification.models import build_model_zoo, predict_scores
from hbb_classification.plots import save_punzi_curves, save_roc_curves
from hbb_classification.split import (
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
    make_feature_label_frames,
    stratified_train_val_test,
)

LOGGER = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        number = float(value)
        if np.isinf(number):
            return None
        return number
    if isinstance(value, float) and np.isinf(value):
        return None
    return value


def _count_sentinel_rows(features: pd.DataFrame) -> dict[str, int]:
    mask = pd.Series(False, index=features.index)
    counts = {}
    for column in SENTINEL_COLUMNS:
        column_mask = features[column] == SENTINEL_VALUE
        counts[column] = int(column_mask.sum())
        mask = mask | column_mask
    counts["any_sentinel_row"] = int(mask.sum())
    return counts


def select_physics_baseline(
    x_val: pd.DataFrame,
    y_val: pd.Series,
    candidates: tuple[str, ...] = PHYSICS_SCORE_CANDIDATES,
) -> dict[str, Any]:
    ranked = []
    for column in candidates:
        raw = x_val[column].to_numpy(dtype=float)
        upright = score_metrics(y_val, raw)
        flipped = score_metrics(y_val, -raw)
        if flipped["roc_auc"] > upright["roc_auc"]:
            ranked.append(
                {
                    "feature": column,
                    "direction": "lower_is_more_signal",
                    "val_roc_auc": flipped["roc_auc"],
                }
            )
        else:
            ranked.append(
                {
                    "feature": column,
                    "direction": "higher_is_more_signal",
                    "val_roc_auc": upright["roc_auc"],
                }
            )
    ranked.sort(key=lambda item: item["val_roc_auc"], reverse=True)
    winner = ranked[0]
    return {"selected": winner, "candidates": ranked}


def physics_scores(features: pd.DataFrame, feature: str, direction: str) -> np.ndarray:
    values = features[feature].to_numpy(dtype=float)
    if direction == "lower_is_more_signal":
        return -values
    return values


def _model_extras(name: str, model: Pipeline) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    if name == "logistic_regression_pca" and "pca" in model.named_steps:
        pca = model.named_steps["pca"]
        extras["n_components"] = int(pca.n_components_)
        extras["explained_variance"] = float(pca.explained_variance_ratio_.sum())
    if "sentinel" in model.named_steps:
        extras["train_sentinel_medians"] = model.named_steps["sentinel"].medians_
        extras["n_sentinel_in_train"] = model.named_steps["sentinel"].n_sentinel_in_fit_
    if name == "random_forest":
        forest = model.named_steps["clf"]
        feature_names = list(model.named_steps["sentinel"].get_feature_names_out())
        importances = pd.Series(forest.feature_importances_, index=feature_names)
        extras["top_features"] = importances.sort_values(ascending=False).head(8).to_dict()
    return extras


def _sentinel_ablation(splits: dict[str, Any], random_state: int) -> dict[str, Any]:
    """Compare logistic regression with and without sentinel treatment (val AUC only)."""
    handled = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            ),
        ]
    )
    # "handled" here is the raw-sentinel control; the main zoo already handles sentinels.
    handled.fit(splits["x_train"], splits["y_train"])
    raw_scores = handled.predict_proba(splits["x_val"])[:, 1]
    return {
        "description": (
            "Logistic regression on raw features that still contain -1 energy-ratio "
            "sentinels, evaluated on validation AUC only."
        ),
        "val_roc_auc_raw_sentinels": score_metrics(splits["y_val"], raw_scores)["roc_auc"],
    }


def run_p1_pipeline(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    figure_dir: Path | None = None,
    skip_random_forest: bool = False,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, Any]:
    features, labels = make_feature_label_frames(frame)
    splits = stratified_train_val_test(
        features,
        labels,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_state=random_state,
    )

    physics_choice = select_physics_baseline(splits["x_val"], splits["y_val"])
    physics_feature = physics_choice["selected"]["feature"]
    physics_direction = physics_choice["selected"]["direction"]
    physics_val_scores = physics_scores(splits["x_val"], physics_feature, physics_direction)
    physics_test_scores = physics_scores(splits["x_test"], physics_feature, physics_direction)
    physics_threshold = best_punzi_threshold(splits["y_val"], physics_val_scores)

    zoo = build_model_zoo(random_state=random_state)
    if skip_random_forest:
        zoo.pop("random_forest")

    model_results: dict[str, Any] = {}
    val_scores: dict[str, np.ndarray] = {
        "physics_baseline": physics_val_scores,
    }
    test_scores: dict[str, np.ndarray] = {
        "physics_baseline": physics_test_scores,
    }
    chosen_thresholds = {"physics_baseline": physics_threshold["threshold"]}

    for name, model in zoo.items():
        LOGGER.info("Fitting %s", name)
        model.fit(splits["x_train"], splits["y_train"])
        _, val_score = predict_scores(model, splits["x_val"])
        _, test_score = predict_scores(model, splits["x_test"])
        punzi = best_punzi_threshold(splits["y_val"], val_score)
        val_eval = evaluate_split(splits["y_val"], val_score, punzi_threshold=punzi["threshold"])
        test_eval = evaluate_split(splits["y_test"], test_score, punzi_threshold=punzi["threshold"])
        model_results[name] = {
            "score_kind": "predict_proba",
            "val": val_eval,
            "test": test_eval,
            "punzi_threshold_from_val": punzi,
            **_model_extras(name, model),
        }
        val_scores[name] = np.asarray(val_score)
        test_scores[name] = np.asarray(test_score)
        chosen_thresholds[name] = punzi["threshold"]

    physics_result = {
        "feature": physics_feature,
        "direction": physics_direction,
        "candidates": physics_choice["candidates"],
        "punzi_threshold_from_val": physics_threshold,
        "val": evaluate_split(
            splits["y_val"],
            physics_val_scores,
            punzi_threshold=physics_threshold["threshold"],
            default_threshold=None,
        ),
        "test": evaluate_split(
            splits["y_test"],
            physics_test_scores,
            punzi_threshold=physics_threshold["threshold"],
            default_threshold=None,
        ),
    }

    val_aucs = {
        "physics_baseline": physics_result["val"]["roc_auc"],
        **{name: block["val"]["roc_auc"] for name, block in model_results.items()},
    }
    selected = max(val_aucs, key=val_aucs.get)

    if figure_dir is not None:
        save_roc_curves(splits["y_test"], test_scores, figure_dir / "roc_test.png")
        save_punzi_curves(
            splits["y_val"],
            val_scores,
            figure_dir / "punzi_validation.png",
            chosen_thresholds=chosen_thresholds,
        )

    handled_val_auc = model_results.get("logistic_regression", {}).get("val", {}).get("roc_auc")
    ablation = _sentinel_ablation(splits, random_state=random_state)
    ablation["val_roc_auc_with_sentinel_handling"] = handled_val_auc

    payload = {
        "methodology": "p1",
        "random_state": random_state,
        "n_rows_raw": int(len(frame)),
        "class_counts_raw": {
            "background": int((labels == 0).sum()),
            "signal": int((labels == 1).sum()),
        },
        "split": splits["summary"],
        "sentinel": {
            "columns": list(SENTINEL_COLUMNS),
            "value": SENTINEL_VALUE,
            "rows": {
                "train": _count_sentinel_rows(splits["x_train"]),
                "val": _count_sentinel_rows(splits["x_val"]),
                "test": _count_sentinel_rows(splits["x_test"]),
            },
            "treatment": "indicator plus train-only median imputation",
            "ablation": ablation,
        },
        "physics_baseline": physics_result,
        "models": model_results,
        "selected_by_validation_auc": selected,
        "validation_auc_ranking": val_aucs,
    }
    return _json_safe(payload)
