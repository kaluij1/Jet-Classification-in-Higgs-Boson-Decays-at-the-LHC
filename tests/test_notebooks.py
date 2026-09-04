"""Notebooks stay valid JSON and import the package rather than retraining."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path("notebooks")


def test_notebooks_are_valid_nbformat() -> None:
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert paths, "expected notebooks/01_eda.ipynb and notebooks/02_results.ipynb"
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        sources = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_p1_pipeline" not in sources
        assert "scripts.train" not in sources or path.name == "02_results.ipynb"


def test_eda_notebook_imports_package() -> None:
    notebook = json.loads((NOTEBOOKS / "01_eda.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "from hbb_classification.data import" in source
    assert "load_raw_csv" in source
