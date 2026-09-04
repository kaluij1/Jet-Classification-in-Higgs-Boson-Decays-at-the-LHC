"""YAML configuration for the training CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs") / "default.yaml"


class ConfigError(ValueError):
    """Raised when a config file is missing required keys or has invalid values."""


@dataclass(frozen=True)
class AppConfig:
    seed: int
    data_path: Path
    check_hash: bool
    train_fraction: float
    val_fraction: float
    test_fraction: float
    metrics_path: Path
    figure_dir: Path
    skip_random_forest: bool


def _require(raw: dict[str, Any], *keys: str) -> Any:
    cursor: Any = raw
    path = []
    for key in keys:
        path.append(key)
        if not isinstance(cursor, dict) or key not in cursor:
            raise ConfigError(f"Config is missing {'.'.join(path)}.")
        cursor = cursor[key]
    return cursor


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping.")

    train = float(_require(raw, "split", "train"))
    val = float(_require(raw, "split", "val"))
    test = float(_require(raw, "split", "test"))
    if abs(train + val + test - 1.0) > 1e-9:
        raise ConfigError("split.train + split.val + split.test must sum to 1.")

    return AppConfig(
        seed=int(_require(raw, "seed")),
        data_path=Path(str(_require(raw, "data", "path"))),
        check_hash=bool(_require(raw, "data", "check_hash")),
        train_fraction=train,
        val_fraction=val,
        test_fraction=test,
        metrics_path=Path(str(_require(raw, "output", "metrics"))),
        figure_dir=Path(str(_require(raw, "output", "figures"))),
        skip_random_forest=bool(_require(raw, "models", "skip_random_forest")),
    )
