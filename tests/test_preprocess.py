import pandas as pd

from hbb_classification.data import SENTINEL_VALUE
from hbb_classification.preprocess import SentinelHandler


def test_sentinel_median_uses_train_only() -> None:
    train = pd.DataFrame(
        {
            "tau_vertexEnergyRatio_0": [0.10, 0.20, 0.30, SENTINEL_VALUE],
            "tau_vertexEnergyRatio_1": [0.40, 0.50, 0.60, SENTINEL_VALUE],
        }
    )
    test = pd.DataFrame(
        {
            "tau_vertexEnergyRatio_0": [SENTINEL_VALUE, 9.9],
            "tau_vertexEnergyRatio_1": [SENTINEL_VALUE, 9.9],
        }
    )
    handler = SentinelHandler()
    handler.fit(train)
    assert handler.medians_["tau_vertexEnergyRatio_0"] == 0.20
    assert handler.medians_["tau_vertexEnergyRatio_1"] == 0.50

    transformed = handler.transform(test)
    assert transformed.loc[0, "tau_vertexEnergyRatio_0"] == 0.20
    assert transformed.loc[0, "tau_vertexEnergyRatio_0_is_sentinel"] == 1.0
    assert transformed.loc[1, "tau_vertexEnergyRatio_0"] == 9.9
    assert transformed.loc[1, "tau_vertexEnergyRatio_0_is_sentinel"] == 0.0
