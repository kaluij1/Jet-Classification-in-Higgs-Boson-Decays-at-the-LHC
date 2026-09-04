"""Load and validate the cms_Hbb.csv extract used by this project."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

EXPECTED_FILENAME = "cms_Hbb.csv"
EXPECTED_SIZE_BYTES = 105_219_354
EXPECTED_SHA256 = "79455ebf15da90a9a28d5f816bee56d428b8799aa78d9439a18dc8119a93c751"
EXPECTED_N_ROWS = 225_868

LABEL_COLUMNS = ("isSignal", "isBackground")
INDEX_COLUMN = "Unnamed: 0"

EXPECTED_COLUMNS = [
    INDEX_COLUMN,
    "jetNTracks",
    "nSV",
    "tau0_trackEtaRel_0",
    "tau0_trackEtaRel_1",
    "tau0_trackEtaRel_2",
    "tau1_trackEtaRel_0",
    "tau1_trackEtaRel_1",
    "tau1_trackEtaRel_2",
    "tau_flightDistance2dSig_0",
    "tau_flightDistance2dSig_1",
    "tau_vertexDeltaR_0",
    "tau_vertexEnergyRatio_0",
    "tau_vertexEnergyRatio_1",
    "tau_vertexMass_0",
    "tau_vertexMass_1",
    "trackSip2dSigAboveBottom_0",
    "trackSip2dSigAboveBottom_1",
    "trackSip2dSigAboveCharm_0",
    "trackSipdSig_0",
    "trackSipdSig_0_0",
    "trackSipdSig_0_1",
    "trackSipdSig_1",
    "trackSipdSig_1_0",
    "trackSipdSig_1_1",
    "trackSipdSig_2",
    "trackSipdSig_3",
    "isBackground",
    "isSignal",
]

FEATURE_COLUMNS = [
    column
    for column in EXPECTED_COLUMNS
    if column not in {INDEX_COLUMN, *LABEL_COLUMNS}
]

SENTINEL_VALUE = -1.0
SENTINEL_COLUMNS = (
    "tau_vertexEnergyRatio_0",
    "tau_vertexEnergyRatio_1",
)

PHYSICS_SCORE_CANDIDATES = (
    "nSV",
    "trackSipdSig_0",
    "trackSip2dSigAboveBottom_0",
    "tau_flightDistance2dSig_0",
)


class DataValidationError(ValueError):
    """Raised when the on-disk CSV does not match the expected extract."""


def default_raw_path() -> Path:
    return Path("data") / "raw" / EXPECTED_FILENAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, *, check_hash: bool = True) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path.resolve()}. "
            "Copy cms_Hbb.csv into data/raw/ or pass --source / --data. "
            "See data/raw/README.md."
        )

    size = path.stat().st_size
    if size != EXPECTED_SIZE_BYTES:
        raise DataValidationError(
            f"{path} is {size:,} bytes; expected {EXPECTED_SIZE_BYTES:,} bytes."
        )

    if check_hash:
        digest = sha256_file(path)
        if digest != EXPECTED_SHA256:
            raise DataValidationError(
                f"{path} SHA-256 is {digest}; expected {EXPECTED_SHA256}."
            )


def validate_schema(frame: pd.DataFrame) -> None:
    actual = list(frame.columns)
    if actual != EXPECTED_COLUMNS:
        raise DataValidationError(
            "CSV columns do not match the expected cms_Hbb.csv schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual}"
        )
    if len(frame) != EXPECTED_N_ROWS:
        raise DataValidationError(
            f"CSV has {len(frame):,} rows; expected {EXPECTED_N_ROWS:,}."
        )


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, FEATURE_COLUMNS]


def load_raw_csv(path: str | Path, *, check_hash: bool = True) -> pd.DataFrame:
    csv_path = Path(path)
    validate_file(csv_path, check_hash=check_hash)
    frame = pd.read_csv(csv_path)
    validate_schema(frame)
    return frame
