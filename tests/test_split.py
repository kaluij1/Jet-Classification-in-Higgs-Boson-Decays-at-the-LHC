import pandas as pd
import pytest

from hbb_classification.split import make_feature_label_frames, stratified_train_val_test


def test_split_sizes_and_stratification(toy_frame: pd.DataFrame) -> None:
    features, labels = make_feature_label_frames(toy_frame)
    splits = stratified_train_val_test(
        features,
        labels,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        random_state=42,
    )
    assert splits["summary"]["n_train"] + splits["summary"]["n_val"] + splits["summary"]["n_test"] == len(
        toy_frame
    )
    for part in ("train", "val", "test"):
        counts = splits["summary"]["class_counts"][part]
        assert counts["signal"] > 0
        assert counts["background"] > 0
    assert "isSignal" not in splits["x_train"].columns
    assert "isBackground" not in splits["x_train"].columns


def test_split_fractions_must_sum_to_one(toy_frame: pd.DataFrame) -> None:
    features, labels = make_feature_label_frames(toy_frame)
    with pytest.raises(ValueError, match="sum"):
        stratified_train_val_test(
            features,
            labels,
            train_fraction=0.5,
            val_fraction=0.2,
            test_fraction=0.2,
        )


def test_feature_matrix_drops_labels(toy_frame: pd.DataFrame) -> None:
    features, labels = make_feature_label_frames(toy_frame)
    assert labels.name == "isSignal"
    assert set(labels.unique()) == {0, 1}
    assert labels.sum() == (toy_frame["isSignal"] == 1).sum()
