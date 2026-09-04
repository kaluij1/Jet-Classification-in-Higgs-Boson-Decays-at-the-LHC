"""Evaluation plots written to artifacts/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from hbb_classification.metrics import punzi_curve


def save_roc_curves(
    y_true,
    scores_by_model: dict[str, np.ndarray],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=name.replace("_", " "))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate (background efficiency)")
    ax.set_ylabel("True positive rate (signal efficiency)")
    ax.set_title("ROC on held-out test set")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_punzi_curves(
    y_true,
    scores_by_model: dict[str, np.ndarray],
    path: Path,
    *,
    chosen_thresholds: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, scores in scores_by_model.items():
        curve = punzi_curve(y_true, scores)
        ax.plot(curve["thresholds"], curve["punzi"], label=name.replace("_", " "))
        if chosen_thresholds and name in chosen_thresholds:
            ax.axvline(chosen_thresholds[name], color="0.5", linestyle=":", linewidth=1)
    ax.set_xlabel("Score threshold")
    ax.set_ylabel(r"Punzi significance  $\varepsilon/(1+\sqrt{B})$")
    ax.set_title("Punzi vs threshold on the validation set")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
