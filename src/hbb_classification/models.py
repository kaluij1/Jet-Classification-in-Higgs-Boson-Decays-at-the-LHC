"""Sklearn model zoo for the P1 leakage-controlled pipeline."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hbb_classification.preprocess import SentinelHandler

PCA_VARIANCE = 0.85


def build_model_zoo(random_state: int = 42) -> dict[str, Pipeline]:
    """Classifiers with train-only preprocessing inside each pipeline."""
    return {
        "naive_bayes": Pipeline(
            [
                ("sentinel", SentinelHandler()),
                ("clf", GaussianNB()),
            ]
        ),
        "lda": Pipeline(
            [
                ("sentinel", SentinelHandler()),
                ("scale", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("sentinel", SentinelHandler()),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logistic_regression_pca": Pipeline(
            [
                ("sentinel", SentinelHandler()),
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=PCA_VARIANCE, random_state=random_state)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("sentinel", SentinelHandler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def predict_scores(model, features) -> tuple[str, object]:
    """Return (score_kind, scores) using predict_proba when available."""
    if hasattr(model, "predict_proba"):
        return "predict_proba", model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return "decision_function", model.decision_function(features)
    return "predict", model.predict(features)
