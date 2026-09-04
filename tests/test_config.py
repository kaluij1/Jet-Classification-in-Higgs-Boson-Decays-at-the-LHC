from pathlib import Path

import pytest

from hbb_classification.config import ConfigError, load_config


def test_default_config_loads() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.seed == 42
    assert config.train_fraction + config.val_fraction + config.test_fraction == pytest.approx(1.0)
    assert config.data_path.as_posix().endswith("cms_Hbb.csv")


def test_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing"):
        load_config(path)


def test_fractions_must_sum_to_one(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
seed: 1
data:
  path: x.csv
  check_hash: false
split:
  train: 0.5
  val: 0.2
  test: 0.2
output:
  metrics: m.json
  figures: figs
models:
  skip_random_forest: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="sum"):
        load_config(path)
