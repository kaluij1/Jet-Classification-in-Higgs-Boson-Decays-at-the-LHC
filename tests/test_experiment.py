import pandas as pd

from hbb_classification.experiment import run_p1_pipeline


def test_p1_smoke_on_toy_frame(toy_frame: pd.DataFrame) -> None:
    results = run_p1_pipeline(
        toy_frame,
        random_state=42,
        figure_dir=None,
        skip_random_forest=True,
    )
    assert results["methodology"] == "p1"
    assert results["split"]["n_train"] + results["split"]["n_val"] + results["split"]["n_test"] == len(
        toy_frame
    )
    assert "logistic_regression" in results["models"]
    assert "random_forest" not in results["models"]
    auc = results["models"]["logistic_regression"]["test"]["roc_auc"]
    assert 0.0 <= auc <= 1.0
    assert results["models"]["logistic_regression"]["test"]["at_punzi_threshold"]["punzi"] >= 0.0
