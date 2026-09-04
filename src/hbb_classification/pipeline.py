"""Reproduce the original notebook training and evaluation path.

This module mirrors the methodology in reports/original/final project.ipynb,
including steps that are scientifically imperfect (undersampling before the
split, PCA fit on the full balanced matrix). Those are left unchanged on
purpose for P0 so the script can match the saved notebook metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from hbb_classification.data import FEATURE_COLUMNS, feature_matrix

RANDOM_STATE = 42
TEST_SIZE = 0.3
PCA_VARIANCE = 0.85
N_RF_FEATURES = 8
N_PUNZI_THRESHOLDS = 200


def undersample_background(frame: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Match the notebook: downsample background on the full frame, then model."""
    signal = frame[frame["isSignal"] == 1]
    background = frame[frame["isSignal"] == 0]
    background_downsampled = resample(
        background,
        replace=False,
        n_samples=len(signal),
        random_state=random_state,
    )
    return pd.concat([signal, background_downsampled])


def _class_block(report: dict[str, Any], label: float) -> dict[str, Any]:
    for key in (str(label), str(int(label)), label):
        if key in report:
            block = report[key]
            return {
                "precision": block["precision"],
                "recall": block["recall"],
                "f1": block["f1-score"],
                "support": int(block["support"]),
            }
    raise KeyError(f"classification_report is missing class {label}: {list(report)}")


def _report_dict(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "background": _class_block(report, 0.0),
        "signal": _class_block(report, 1.0),
    }


def punzi_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_thresholds: int = N_PUNZI_THRESHOLDS,
) -> dict[str, Any]:
    """Notebook Punzi: epsilon / (1 + sqrt(B)); 0 when B == 0."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    scores = []
    n_signal = np.sum(y_true == 1)
    for threshold in thresholds:
        predicted_signal = y_proba > threshold
        epsilon = np.sum((predicted_signal == 1) & (y_true == 1)) / n_signal
        n_background = np.sum((predicted_signal == 1) & (y_true == 0))
        if n_background == 0:
            scores.append(0.0)
        else:
            scores.append(float(epsilon / (1.0 + np.sqrt(n_background))))
    scores_array = np.asarray(scores)
    best_index = int(np.argmax(scores_array))
    return {
        "best_threshold": float(thresholds[best_index]),
        "best_score": float(scores_array[best_index]),
    }


def run_notebook_pipeline(
    frame: pd.DataFrame,
    *,
    random_state: int = RANDOM_STATE,
    skip_random_forest: bool = False,
) -> dict[str, Any]:
    balanced = undersample_background(frame, random_state=random_state)
    features = feature_matrix(balanced)
    labels = balanced["isSignal"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=labels,
    )

    models = {
        "naive_bayes": GaussianNB(),
        "lda": LinearDiscriminantAnalysis(),
        "logistic_regression": LogisticRegression(max_iter=1000),
    }
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        predictions[name] = pred
        metrics[name] = _report_dict(y_test, pred)

    print("Fitting PCA logistic regression...")
    # Same leakage as the notebook: scale and PCA on the full balanced X.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=PCA_VARIANCE)
    x_pca = pca.fit_transform(x_scaled)
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(
        x_pca,
        labels,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=labels,
    )
    logreg_pca = LogisticRegression(max_iter=1000)
    logreg_pca.fit(x_train_pca, y_train_pca)
    pred_pca = logreg_pca.predict(x_test_pca)
    metrics["logistic_regression_pca"] = _report_dict(y_test_pca, pred_pca)
    metrics["logistic_regression_pca"]["n_components"] = int(pca.n_components_)
    metrics["logistic_regression_pca"]["explained_variance"] = float(
        pca.explained_variance_ratio_.sum()
    )

    subset_features: list[str] = []
    if not skip_random_forest:
        print("Fitting random forest for feature ranking...")
        forest = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
        forest.fit(x_train, y_train)
        importances = pd.Series(forest.feature_importances_, index=x_train.columns)
        subset_features = importances.sort_values(ascending=False).head(N_RF_FEATURES).index.tolist()
        x_subset = balanced[subset_features]
        x_train_sub, x_test_sub, y_train_sub, y_test_sub = train_test_split(
            x_subset,
            labels,
            test_size=TEST_SIZE,
            random_state=random_state,
            stratify=labels,
        )
        logreg_sub = LogisticRegression(max_iter=1000)
        logreg_sub.fit(x_train_sub, y_train_sub)
        metrics["logistic_regression_rf_top8"] = _report_dict(
            y_test_sub, logreg_sub.predict(x_test_sub)
        )
        metrics["logistic_regression_rf_top8"]["features"] = subset_features

    logreg = models["logistic_regression"]
    y_proba = logreg.predict_proba(x_test)[:, 1]
    punzi_all = punzi_curve(y_test.to_numpy(), y_proba)
    y_proba_pca = logreg_pca.predict_proba(x_test_pca)[:, 1]
    punzi_pca = punzi_curve(y_test_pca.to_numpy(), y_proba_pca)

    return {
        "n_rows_raw": int(len(frame)),
        "n_rows_balanced": int(len(balanced)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features": len(FEATURE_COLUMNS),
        "class_counts_raw": {
            "background": int((frame["isSignal"] == 0).sum()),
            "signal": int((frame["isSignal"] == 1).sum()),
        },
        "metrics": metrics,
        "punzi_all_features": punzi_all,
        "punzi_pca": punzi_pca,
        "rf_top8_features": subset_features,
    }
